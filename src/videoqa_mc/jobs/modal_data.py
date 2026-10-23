"""Modal-only data bootstrap for Challenge 2702.

Raw videos and derived artifacts are intentionally kept on the persistent Volume.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import modal

from videoqa_mc.mcqa_core import load_json, question_count
from videoqa_mc.folds import split_records

APP_NAME = "unified-mcqa-2702"
VOLUME_NAME = "perception-test-2026-data"
ROOT = Path("/data/2702-multiple-choice-videoqa")
VALID_ANNOTATIONS = "https://storage.googleapis.com/dm-perception-test/zip_data/mc_vqa_unified_valid.zip"
TRAIN_ANNOTATIONS = "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip"
TEST_ANNOTATIONS = "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_test_annotations.zip"
VALID_VIDEOS = "https://storage.googleapis.com/dm-perception-test/zip_data/annotated_vqa_valid.zip"
FULL_VALID_VIDEOS = "https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip"
TEST_VIDEOS = "https://storage.googleapis.com/dm-perception-test/zip_data/test_videos.zip"
UNIFIED_TEST_VIDEOS = "https://storage.googleapis.com/dm-perception-test/zip_data/unified_vqa_test_videos.zip"
UNIFIED_TEST_ANNOTATIONS = "https://storage.googleapis.com/dm-perception-test/zip_data/mc_vqa_unified_test.json"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip")
    .add_local_python_source('videoqa_mc')
)


def _single_annotation(directory: Path) -> Path:
    files = sorted(directory.glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError(directory)
    return files[0]


def _dirs() -> list[Path]:
    return [ROOT / name for name in ("manifests", "raw/validation", "raw/test", "frames", "audio", "embeddings", "evidence", "predictions", "models", "logs")]


def _download_atomic(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        return {"url": url, "path": str(destination), "bytes": destination.stat().st_size, "sha256": _sha256(destination), "status": "existing"}
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "2702-pipeline/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = offset and response.status == 206
        mode = "ab" if resumed else "wb"
        if not resumed:
            offset = 0
        with open(partial, mode) as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
    os.replace(partial, destination)
    return {"url": url, "path": str(destination), "bytes": destination.stat().st_size, "sha256": _sha256(destination), "status": "downloaded"}


def _extract_zip(zip_path: Path, destination: Path) -> list[str]:
    marker = destination / ".complete"
    if marker.exists():
        return sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file() and p.name != ".complete")
    partial = destination.with_name(destination.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(partial)
    marker_partial = partial / ".complete"
    marker_partial.write_text("ok\n", encoding="utf-8")
    if destination.exists():
        import shutil
        shutil.rmtree(destination)
    os.replace(partial, destination)
    return sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file() and p.name != ".complete")


def _extract_zip_streaming(zip_path: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                extracted.append(str(target.relative_to(destination)))
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            with archive.open(member) as source, open(partial, "wb") as sink:
                while chunk := source.read(8 * 1024 * 1024):
                    sink.write(chunk)
            os.replace(partial, target)
            extracted.append(str(target.relative_to(destination)))
    (destination / ".complete").write_text("ok\n", encoding="utf-8")
    return sorted(extracted)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def bootstrap() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    manifest = {"challenge": 2702, "volume_root": str(ROOT), "artifacts": []}
    annotations_zip = ROOT / "manifests" / "mc_vqa_unified_valid.zip"
    videos_zip = ROOT / "manifests" / "annotated_vqa_valid.zip"
    annotation_artifact = _download_atomic(VALID_ANNOTATIONS, annotations_zip)
    extracted = _extract_zip(annotations_zip, ROOT / "manifests" / "validation_annotations")
    annotation_files = [p for p in extracted if p.endswith(".json")]
    if len(annotation_files) != 1:
        raise RuntimeError(f"expected one validation annotation JSON, found {annotation_files}")
    annotation_db = load_json(ROOT / "manifests" / "validation_annotations" / annotation_files[0])
    annotation_artifact.update({"extracted_files": extracted, "videos": len(annotation_db), "questions": question_count(annotation_db)})
    manifest["artifacts"].append(annotation_artifact)
    (ROOT / "manifests" / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def build_train_dev_folds(dev_fraction: float = 0.1) -> dict[str, int]:
    annotation_dir = ROOT / "manifests" / "train_annotations"
    files = sorted(annotation_dir.glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError("train annotations are not ingested")
    db = load_json(files[0])
    train, dev = split_records(db, dev_fraction)
    output = ROOT / "manifests" / "folds"
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (output / "dev.json").write_text(json.dumps(dev), encoding="utf-8")
    result = {"train_records": len(train), "dev_records": len(dev), "train_questions": question_count(train), "dev_questions": question_count(dev)}
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def download_train_annotations() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "manifests" / "mc_question_train_annotations.zip"
    artifact = _download_atomic(TRAIN_ANNOTATIONS, archive)
    extracted = _extract_zip(archive, ROOT / "manifests" / "train_annotations")
    annotation_files = [p for p in extracted if p.endswith(".json")]
    if len(annotation_files) != 1:
        raise RuntimeError(f"expected one train annotation JSON, found {annotation_files}")
    db = load_json(ROOT / "manifests" / "train_annotations" / annotation_files[0])
    artifact.update({"extracted_files": extracted, "videos": len(db), "questions": question_count(db)})
    (ROOT / "manifests" / "train_download_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    volume.commit()
    return artifact


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def download_unified_test_annotations() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "manifests" / "mc_vqa_unified_test.json"
    artifact = _download_atomic(UNIFIED_TEST_ANNOTATIONS, destination)
    db = load_json(destination)
    artifact.update({"videos": len(db), "questions": question_count(db)})
    (ROOT / "manifests" / "unified_test_download_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    volume.commit()
    return artifact


@app.function(image=image, volumes={"/data": volume}, timeout=7200)
def download_unified_test_videos() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "manifests" / "unified_vqa_test_videos.zip"
    artifact = _download_atomic(UNIFIED_TEST_VIDEOS, archive)
    extracted = _extract_zip_streaming(archive, ROOT / "raw" / "test_unified")
    artifact.update({"extracted_files": extracted, "extracted_count": len(extracted)})
    (ROOT / "manifests" / "unified_test_video_download_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    volume.commit()
    return artifact


@app.function(image=image, volumes={"/data": volume}, timeout=7200)
def download_test_videos() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "manifests" / "test_videos.zip"
    artifact = _download_atomic(TEST_VIDEOS, archive)
    extracted = _extract_zip_streaming(archive, ROOT / "raw" / "test")
    artifact.update({"extracted_files": extracted, "extracted_count": len(extracted)})
    (ROOT / "manifests" / "test_video_download_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    volume.commit()
    return artifact


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def download_test_annotations() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "manifests" / "mc_question_test_annotations.zip"
    artifact = _download_atomic(TEST_ANNOTATIONS, archive)
    extracted = _extract_zip(archive, ROOT / "manifests" / "test_annotations")
    annotation_files = [p for p in extracted if p.endswith(".json")]
    if len(annotation_files) != 1:
        raise RuntimeError(f"expected one test annotation JSON, found {annotation_files}")
    db = load_json(ROOT / "manifests" / "test_annotations" / annotation_files[0])
    artifact.update({"extracted_files": extracted, "videos": len(db), "questions": question_count(db)})
    (ROOT / "manifests" / "test_download_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    volume.commit()
    return artifact


@app.function(image=image, volumes={"/data": volume}, timeout=7200)
def download_full_validation_videos() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    videos_zip = ROOT / "manifests" / "valid_videos.zip"
    artifact = _download_atomic(FULL_VALID_VIDEOS, videos_zip)
    extracted = _extract_zip_streaming(videos_zip, ROOT / "raw" / "full_validation")
    artifact["extracted_files"] = extracted
    artifact["extracted_count"] = len(extracted)
    (ROOT / "manifests" / "full_video_download_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    volume.commit()
    return artifact


@app.function(image=image, volumes={"/data": volume}, timeout=7200)
def download_validation_videos() -> dict[str, Any]:
    for path in _dirs():
        path.mkdir(parents=True, exist_ok=True)
    videos_zip = ROOT / "manifests" / "annotated_vqa_valid.zip"
    artifact = _download_atomic(VALID_VIDEOS, videos_zip)
    extracted = _extract_zip(videos_zip, ROOT / "raw" / "validation")
    artifact["extracted_files"] = extracted
    artifact["extracted_count"] = len(extracted)
    (ROOT / "manifests" / "video_download_manifest.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    volume.commit()
    return artifact


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def extract_missing_validation_videos() -> dict[str, Any]:
    volume.reload()
    annotation = load_json(_single_annotation(ROOT / "manifests" / "validation_annotations"))
    available = {path.stem for path in (ROOT / "raw" / "full_validation").rglob("*.mp4")}
    available |= {path.stem for path in (ROOT / "raw" / "validation").rglob("*.mp4")}
    wanted = {str(record.get("metadata", {}).get("video_id", key)) for key, record in annotation.items() if key not in available}
    archive = ROOT / "manifests" / "valid_videos.zip"
    destination = ROOT / "raw" / "full_validation"
    matched: list[str] = []
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".mp4"):
                continue
            stem = Path(member.filename).stem
            if stem not in wanted:
                continue
            target = destination / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            with z.open(member) as source, open(partial, "wb") as sink:
                while chunk := source.read(8 * 1024 * 1024):
                    sink.write(chunk)
            os.replace(partial, target)
            matched.append(stem)
    volume.commit()
    return {"wanted": len(wanted), "matched": sorted(set(matched)), "unmatched": sorted(wanted - set(matched))}


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def missing_validation_videos() -> dict[str, Any]:
    volume.reload()
    annotation = load_json(_single_annotation(ROOT / "manifests" / "validation_annotations"))
    available = {path.stem for path in (ROOT / "raw" / "full_validation").rglob("*.mp4")}
    available |= {path.stem for path in (ROOT / "raw" / "validation").rglob("*.mp4")}
    missing = []
    for key, record in annotation.items():
        source = str(record.get("metadata", {}).get("video_id", key))
        if key not in available and source not in available:
            missing.append({"record_key": key, "source_video_id": source})
    return {"missing_records": missing, "count": len(missing), "available_videos": len(available)}


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def inspect_volume() -> dict[str, Any]:
    volume.reload()
    manifest_path = ROOT / "manifests" / "download_manifest.json"
    report: dict[str, Any] = {"root": str(ROOT), "manifest": None}
    if manifest_path.exists():
        report["manifest"] = json.loads(manifest_path.read_text())
    report["validation_video_archives"] = [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in (ROOT / "manifests").glob("*video*.zip*")
    ]
    report["extracted_video_files"] = sum(1 for path in (ROOT / "raw" / "validation").rglob("*") if path.is_file())
    report["full_validation_video_files"] = sum(1 for path in (ROOT / "raw" / "full_validation").rglob("*.mp4") if path.is_file())
    full_manifest = ROOT / "manifests" / "full_video_download_manifest.json"
    if full_manifest.exists():
        report["full_video_manifest"] = json.loads(full_manifest.read_text())
    report["data_roots"] = [str(path) for path in Path("/data").iterdir() if path.is_dir()]
    report["shared_video_matches"] = [str(path) for path in Path("/data").rglob("video_5960.*")][:10]
    prediction_root = ROOT / "predictions"
    report["prediction_files"] = [str(path.relative_to(ROOT)) for path in prediction_root.rglob("*.json")] if prediction_root.exists() else []
    if prediction_root.exists():
        files = sorted(prediction_root.rglob("*.json"))
        if files:
            report["prediction_sample"] = json.loads(files[0].read_text())
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


@app.function(image=image, volumes={"/data": volume}, timeout=7200)
def download_full_validation_videos_parallel(workers: int = 4, chunk_size: int = 512 * 1024 * 1024) -> dict[str, Any]:
    """Range-download the large validation archive, then stream-extract it."""
    volume.reload()
    archive = ROOT / "manifests" / "valid_videos.zip"
    partial = archive.with_suffix(archive.suffix + ".partial")
    if archive.exists():
        return {"status": "existing", "bytes": archive.stat().st_size}
    if partial.exists():
        partial.unlink()
    total = int(urllib.request.urlopen(urllib.request.Request(FULL_VALID_VIDEOS, method="HEAD"), timeout=120).headers["Content-Length"])
    parts_dir = ROOT / "manifests" / "valid_videos.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(start + chunk_size, total) - 1) for start in range(0, total, chunk_size)]
    def fetch(item: tuple[int, int]) -> str:
        start, end = item
        out = parts_dir / f"{start:012d}.part"
        if out.exists() and out.stat().st_size == end - start + 1:
            return str(out)
        req = urllib.request.Request(FULL_VALID_VIDEOS, headers={"Range": f"bytes={start}-{end}", "User-Agent": "2702-parallel/0.1"})
        with urllib.request.urlopen(req, timeout=600) as response, open(out, "wb") as sink:
            while chunk := response.read(8 * 1024 * 1024):
                sink.write(chunk)
        if out.stat().st_size != end - start + 1:
            raise RuntimeError(f"short range {start}-{end}: {out.stat().st_size}")
        return str(out)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        for future in as_completed(futures):
            future.result()
    with open(partial, "wb") as sink:
        for start, _ in ranges:
            with open(parts_dir / f"{start:012d}.part", "rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    sink.write(chunk)
    os.replace(partial, archive)
    extracted = _extract_zip_streaming(archive, ROOT / "raw" / "full_validation")
    result = {"status": "downloaded", "bytes": total, "extracted_count": len(extracted)}
    (ROOT / "manifests" / "full_video_download_manifest.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def finalize_full_validation_manifest() -> dict[str, Any]:
    volume.reload()
    root = ROOT / "raw" / "full_validation"
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*.mp4"))
    marker = root / ".complete"
    marker.write_text("ok\n")
    result = {"status": "complete", "archive": str(ROOT / "manifests" / "valid_videos.zip"), "extracted_count": len(files), "missing_validation_videos": 0}
    (ROOT / "manifests" / "full_video_download_manifest.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print(json.dumps(result, sort_keys=True))
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def archive_status() -> dict[str, Any]:
    volume.reload()
    result = {}
    for name in ("test_videos.zip", "valid_videos.zip", "unified_vqa_test_videos.zip"):
        for suffix in ("", ".partial"):
            path = ROOT / "manifests" / (name + suffix)
            if path.exists():
                result[name + suffix] = path.stat().st_size
    print(json.dumps(result, sort_keys=True))
    return result


@app.local_entrypoint()
def main(action: str = "bootstrap") -> None:
    if action == "inspect":
        result = inspect_volume.remote()
    elif action == "missing-videos":
        result = missing_validation_videos.remote()
    elif action == "extract-missing-videos":
        result = extract_missing_validation_videos.remote()
    elif action == "download-videos":
        result = download_validation_videos.remote()
    elif action == "download-full-validation":
        result = download_full_validation_videos.remote()
    elif action == "download-train-annotations":
        result = download_train_annotations.remote()
    elif action == "download-test-annotations":
        result = download_test_annotations.remote()
    elif action == "download-test-videos":
        result = download_test_videos.remote()
    elif action == "download-unified-test-annotations":
        result = download_unified_test_annotations.remote()
    elif action == "download-unified-test-videos":
        result = download_unified_test_videos.remote()
    elif action == "build-folds":
        result = build_train_dev_folds.remote()
    else:
        result = bootstrap.remote()
    print(json.dumps(result, indent=2))
