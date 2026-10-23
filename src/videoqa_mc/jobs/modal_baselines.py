"""Label-based baselines used only for train/dev diagnostics and one external val run."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import modal

from videoqa_mc.mcqa_core import make_answer, question_count, validate_predictions

ROOT = Path("/data/2702-multiple-choice-videoqa")
VOLUME_NAME = "perception-test-2026-data"
app = modal.App("unified-mcqa-2702-baselines")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").add_local_python_source('videoqa_mc')


def _single_json(directory: Path) -> Path:
    files = sorted(directory.glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError(directory)
    return files[0]


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def frequency_validation() -> dict[str, Any]:
    volume.reload()
    train = json.loads(_single_json(ROOT / "manifests" / "train_annotations").read_text())
    valid = json.loads(_single_json(ROOT / "manifests" / "validation_annotations").read_text())
    exact: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for record in train.values():
        for question in record.get("mc_question") or []:
            answer = question["options"][question["answer_id"]]
            exact[question["question"]][answer] += 1
            global_counts[answer] += 1
    predictions: dict[str, list[dict[str, Any]]] = {}
    for key, record in valid.items():
        answers = []
        for question in record.get("mc_question") or []:
            counts = exact.get(question["question"], Counter())
            chosen = max(question["options"], key=lambda option: (counts[option], global_counts[option]))
            answers.append(make_answer(question, question["options"].index(chosen)))
        predictions[key] = answers
    errors = validate_predictions(predictions, valid)
    correct = total = 0
    for key, record in valid.items():
        for question, answer in zip(record.get("mc_question") or [], predictions[key]):
            total += 1
            correct += int(answer["answer_id"] == question["answer_id"])
    output = ROOT / "predictions" / "validation" / "frequency_submission.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    volume.commit()
    return {"errors": errors, "questions": total, "correct": correct, "top1": correct / total if total else 0.0, "output": str(output)}


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def evaluate_cached(name: str = "qwen_submission.json") -> dict[str, Any]:
    volume.reload()
    valid = json.loads(_single_json(ROOT / "manifests" / "validation_annotations").read_text())
    submission_path = ROOT / "predictions" / "validation" / name
    if name.startswith("molmo_fast"):
        predictions = {}
        for path in sorted((ROOT / "predictions" / "validation").glob("molmo_fast_video_*.json")):
            result = json.loads(path.read_text())
            predictions.setdefault(result["record_key"], []).append(result["answer"])
    elif name.startswith("molmo_final"):
        predictions = {}
        for path in sorted((ROOT / "predictions" / "validation").glob("molmo_final_video_*.json")):
            result = json.loads(path.read_text())
            predictions.setdefault(result["record_key"], []).append(result["answer"])
    elif name.startswith("molmo"):
        predictions = {}
        for path in sorted((ROOT / "predictions" / "validation").glob("molmo_video_*.json")):
            result = json.loads(path.read_text())
            predictions.setdefault(result["record_key"], []).append(result["answer"])
    else:
        predictions = json.loads(submission_path.read_text())
    total = correct = 0
    for key, answers in predictions.items():
        for answer in answers:
            question = next(q for q in valid[key]["mc_question"] if int(q["id"]) == int(answer["id"]))
            total += 1
            correct += int(answer["answer_id"] == question["answer_id"])
    expected = sum(len(record.get("mc_question") or []) for record in valid.values())
    return {"name": name, "correct": correct, "scored": total, "expected": expected, "coverage": total / expected, "partial_top1": correct / total if total else 0.0}


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def build_test_submission() -> dict[str, Any]:
    volume.reload()
    original = json.loads((ROOT / "manifests" / "test_annotations" / "mc_question_test.json").read_text())
    additional = json.loads((ROOT / "manifests" / "mc_vqa_unified_test.json").read_text())
    predictions: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for split, annotations, directory in (("original", original, ROOT / "predictions" / "test"), ("additional", additional, ROOT / "predictions" / "test_unified")):
        files = {path.name: path for path in directory.glob("molmo_*.json")}
        for key, record in annotations.items():
            answers = []
            for index, question in enumerate(record.get("mc_question") or []):
                path = files.get(f"molmo_{key}_{index}.json")
                if path is None:
                    missing.append(f"{split}:{key}:{index}")
                    continue
                result = json.loads(path.read_text())
                answers.append(result["answer"])
            if answers:
                predictions[key] = answers
    expected = question_count(original) + question_count(additional)
    scored = sum(len(v) for v in predictions.values())
    if missing:
        raise RuntimeError(f"missing {len(missing)} predictions; first={missing[:5]}")
    errors = validate_predictions(predictions, {**original, **additional})
    if errors:
        raise RuntimeError(f"invalid submission: {errors[:5]}")
    output = ROOT / "predictions" / "test_submission.json"
    output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    manifest = {"original_questions": question_count(original), "additional_questions": question_count(additional), "questions": scored, "expected": expected, "coverage": scored / expected if expected else 0.0, "errors": errors, "model": "allenai/Molmo2-8B"}
    (ROOT / "predictions" / "test_submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def build_test_chunk(split: str, record_start: int, record_end: int) -> dict[str, Any]:
    volume.reload()
    if split == "additional":
        annotations = json.loads((ROOT / "manifests" / "mc_vqa_unified_test.json").read_text())
        directory = ROOT / "predictions" / "test_unified"
    else:
        annotations = json.loads((ROOT / "manifests" / "test_annotations" / "mc_question_test.json").read_text())
        directory = ROOT / "predictions" / "test"
    result: dict[str, list[dict[str, Any]]] = {}
    missing = 0
    for key, record in list(annotations.items())[record_start:record_end]:
        answers = []
        for index, question in enumerate(record.get("mc_question") or []):
            path = directory / f"molmo_{key}_{index}.json"
            if not path.exists():
                missing += 1
                continue
            answers.append(json.loads(path.read_text())["answer"])
        if answers:
            result[key] = answers
    out = ROOT / "predictions" / f"{split}_submission_{record_start}_{record_end}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    volume.commit()
    return {"split": split, "record_start": record_start, "record_end": record_end, "records": len(result), "questions": sum(map(len, result.values())), "missing": missing}


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def merge_test_chunks() -> dict[str, Any]:
    volume.reload()
    paths = sorted((ROOT / "predictions").glob("original_submission_*.json")) + sorted((ROOT / "predictions").glob("additional_submission_*.json"))
    predictions: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        chunk = json.loads(path.read_text())
        overlap = set(predictions).intersection(chunk)
        if overlap:
            raise RuntimeError(f"duplicate records across chunks: {sorted(overlap)[:5]}")
        predictions.update(chunk)
    original = json.loads((ROOT / "manifests" / "test_annotations" / "mc_question_test.json").read_text())
    additional = json.loads((ROOT / "manifests" / "mc_vqa_unified_test.json").read_text())
    annotations = {**original, **additional}
    errors = validate_predictions(predictions, annotations)
    expected = question_count(annotations)
    scored = sum(len(v) for v in predictions.values())
    if errors or scored != expected:
        raise RuntimeError({"errors": errors[:10], "scored": scored, "expected": expected})
    output = ROOT / "predictions" / "test_submission.json"
    output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    manifest = {"original_questions": question_count(original), "additional_questions": question_count(additional), "questions": scored, "expected": expected, "coverage": scored / expected, "errors": errors, "model": "allenai/Molmo2-8B"}
    (ROOT / "predictions" / "test_submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def evaluate_molmo_breakdown(prefix: str = "molmo_final") -> dict[str, Any]:
    volume.reload()
    valid = json.loads(_single_json(ROOT / "manifests" / "validation_annotations").read_text())
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path in (ROOT / "predictions" / "validation").glob(f"{prefix}_video_*.json"):
        result = json.loads(path.read_text())
        key = result["record_key"]
        answer = result["answer"]
        q = next(q for q in valid[key]["mc_question"] if int(q["id"]) == int(answer["id"]))
        group = "extra_5choice" if len(q["options"]) == 5 else "original_3choice"
        groups[group][1] += 1
        groups[group][0] += int(answer["answer_id"] == q["answer_id"])
    return {k: {"correct": v[0], "scored": v[1], "accuracy": v[0] / v[1] if v[1] else 0.0} for k, v in groups.items()}


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def build_special_test_submission() -> dict[str, Any]:
    volume.reload()
    base = json.loads((ROOT / "predictions" / "test_submission.json").read_text())
    additional = json.loads((ROOT / "manifests" / "mc_vqa_unified_test.json").read_text())
    directory = ROOT / "predictions" / "test_unified_special"
    for key, record in additional.items():
        answers = []
        for index, question in enumerate(record.get("mc_question") or []):
            path = directory / f"molmo_{key}_{index}.json"
            if not path.exists():
                raise RuntimeError(f"missing {key}:{index}")
            answers.append(json.loads(path.read_text())["answer"])
        base[key] = answers
    original = json.loads((ROOT / "manifests" / "test_annotations" / "mc_question_test.json").read_text())
    errors = validate_predictions(base, {**original, **additional})
    expected = question_count(original) + question_count(additional)
    scored = sum(len(v) for v in base.values())
    if errors or scored != expected:
        raise RuntimeError({"errors": errors[:10], "scored": scored, "expected": expected})
    output = ROOT / "predictions" / "test_submission_special.json"
    output.write_text(json.dumps(base, indent=2), encoding="utf-8")
    manifest = {"questions": scored, "expected": expected, "coverage": scored / expected, "errors": errors, "model": "allenai/Molmo2-8B specialist prompt"}
    (ROOT / "predictions" / "test_submission_special_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


@app.function(image=image, volumes={"/data": volume}, timeout=600)
def build_gemini_additional_submission() -> dict[str, Any]:
    """Keep the frozen prior original answers and replace only additional answers."""
    volume.reload()
    base_path = ROOT / "predictions" / "test_submission_tomorrow_candidate.json"
    base = json.loads(base_path.read_text())
    original = json.loads((ROOT / "manifests" / "test_annotations" / "mc_question_test.json").read_text())
    additional = json.loads((ROOT / "manifests" / "mc_vqa_unified_test.json").read_text())
    original_before = {key: base[key] for key in original}
    directory = ROOT / "predictions" / "test_unified_gemini"
    prefix = "gemini37_test_whitestjohn_adaptive"
    missing = []
    for key, record in additional.items():
        answers = []
        for index, _question in enumerate(record.get("mc_question") or []):
            path = directory / f"{prefix}_{key}_{index}.json"
            if not path.exists():
                missing.append(f"{key}:{index}")
                continue
            answers.append(json.loads(path.read_text())["answer"])
        if len(answers) == len(record.get("mc_question") or []):
            base[key] = answers
    if missing:
        raise RuntimeError(f"missing {len(missing)} Gemini predictions; first={missing[:10]}")
    if {key: base[key] for key in original} != original_before:
        raise RuntimeError("original-set answers changed")
    annotations = {**original, **additional}
    errors = validate_predictions(base, annotations)
    expected = question_count(annotations)
    scored = sum(len(v) for v in base.values())
    if errors or scored != expected or set(base) != set(annotations):
        raise RuntimeError({"errors": errors[:10], "scored": scored, "expected": expected})
    output = ROOT / "predictions" / "test_submission_gemini_additional.json"
    output.write_text(json.dumps(base, indent=2), encoding="utf-8")
    manifest = {
        "original_questions": question_count(original),
        "additional_questions": question_count(additional),
        "questions": scored,
        "expected": expected,
        "coverage": scored / expected,
        "errors": errors,
        "original_source": str(base_path),
        "additional_model": "gemini-3.7-flash",
        "additional_prefix": prefix,
        "output": str(output),
    }
    (ROOT / "predictions" / "test_submission_gemini_additional_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    volume.commit()
    return manifest

@app.function(image=image, volumes={"/data": volume}, timeout=3600)
def build_full_gemini_submission() -> dict[str, Any]:
    """Replace both sets with complete Gemini checkpoints, retaining prior shape."""
    volume.reload()
    base = json.loads((ROOT / "predictions" / "test_submission_tomorrow_candidate.json").read_text())
    original = json.loads((ROOT / "manifests" / "test_annotations" / "mc_question_test.json").read_text())
    additional = json.loads((ROOT / "manifests" / "mc_vqa_unified_test.json").read_text())
    dirs = ((original, ROOT / "predictions" / "test_gemini", "gemini37_test_whitestjohn_adaptive"),
            (additional, ROOT / "predictions" / "test_unified_gemini", "gemini37_test_whitestjohn_adaptive"))
    missing = []
    for db, directory, prefix in dirs:
        for key, record in db.items():
            answers = []
            for index, _question in enumerate(record.get("mc_question") or []):
                path = directory / f"{prefix}_{key}_{index}.json"
                if not path.exists():
                    missing.append(f"{key}:{index}")
                else:
                    answers.append(json.loads(path.read_text())["answer"])
            if len(answers) == len(record.get("mc_question") or []):
                base[key] = answers
    if missing:
        raise RuntimeError(f"missing {len(missing)} Gemini predictions; first={missing[:10]}")
    annotations = {**original, **additional}
    errors = validate_predictions(base, annotations)
    expected = question_count(annotations)
    scored = sum(len(v) for v in base.values())
    if errors or scored != expected or set(base) != set(annotations):
        raise RuntimeError({"errors": errors[:10], "scored": scored, "expected": expected})
    output = ROOT / "predictions" / "test_submission_full_gemini.json"
    output.write_text(json.dumps(base, indent=2), encoding="utf-8")
    manifest = {"original_questions": question_count(original), "additional_questions": question_count(additional),
                "questions": scored, "expected": expected, "coverage": scored / expected, "errors": errors,
                "original_model": "gemini-3.7-flash", "additional_model": "gemini-3.7-flash", "output": str(output)}
    (ROOT / "predictions" / "test_submission_full_gemini_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(action: str = "frequency") -> None:
    if action == "evaluate-qwen":
        result = evaluate_cached.remote("qwen_submission.json")
    elif action == "evaluate-molmo":
        result = evaluate_cached.remote("molmo_submission.json")
    elif action == "evaluate-molmo-fast":
        result = evaluate_cached.remote("molmo_fast_submission.json")
    elif action == "evaluate-molmo-final":
        result = evaluate_cached.remote("molmo_final_submission.json")
    elif action == "build-test-submission":
        result = build_test_submission.remote()
    elif action == "merge-test-chunks":
        result = merge_test_chunks.remote()
    elif action == "breakdown":
        result = evaluate_molmo_breakdown.remote()
    elif action == "breakdown-special":
        result = evaluate_molmo_breakdown.remote("molmo_special")
    elif action == "build-special-test-submission":
        result = build_special_test_submission.remote()
    else:
        result = frequency_validation.remote()
    print(json.dumps(result, indent=2))
