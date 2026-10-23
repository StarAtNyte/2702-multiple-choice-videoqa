"""Question-directed Qwen baseline for Challenge 2702.

Inference consumes only question/options plus video frames. Outputs are cached
per question so long runs can resume after interruption.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal

from videoqa_mc.mcqa_core import make_answer, validate_predictions
from videoqa_mc.question_planner import classify

APP_NAME = "unified-mcqa-2702-inference"
VOLUME_NAME = "perception-test-2026-data"
ROOT = Path("/data/2702-multiple-choice-videoqa")
MODEL_NAME = os.getenv("MCQA_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "ca-certificates")
    .pip_install(
        "accelerate>=1.9,<2", "pillow>=11,<13", "qwen-vl-utils>=0.0.14,<1",
        "torch==2.7.1", "torchvision==0.22.1", "transformers>=4.57,<5",
    )
    .env({"HF_HOME": str(ROOT / "models" / "huggingface")})
    .add_local_python_source('videoqa_mc')
)


def _annotation_path() -> Path:
    candidates = sorted((ROOT / "manifests" / "validation_annotations").glob("*.json"))
    if len(candidates) != 1:
        raise FileNotFoundError("validation annotation JSON is not ingested")
    return candidates[0]


def _find_video(*identifiers: str) -> Path:
    roots = [ROOT / "raw" / "validation", ROOT / "raw" / "full_validation", ROOT / "manifests", Path("/data/shared")]
    for identifier in identifiers:
        matches = [p for root in roots if root.exists() for p in root.rglob(f"{identifier}.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
        if not matches:
            matches = [p for root in roots if root.exists() for p in root.rglob(f"{identifier}_*.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
        if matches:
            return matches[0]
    raise FileNotFoundError(f"video not found for identifiers: {identifiers}")


def _sample_frames(video: Path, count: int, output: Path) -> list[tuple[float, Path]]:
    output.mkdir(parents=True, exist_ok=True)
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)], text=True).strip())
    timestamps = [0.0] if duration <= 0 else [duration * i / max(1, count - 1) for i in range(count)]
    rows = []
    for index, timestamp in enumerate(timestamps):
        path = output / f"{index:03d}.jpg"
        if not path.exists():
            safe_timestamp = min(timestamp, max(0.0, duration - 0.05))
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{safe_timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=768:-2", "-q:v", "3", str(path)], check=True)
        if path.exists():
            rows.append((timestamp, path))
    if not rows:
        raise RuntimeError(f"No frames extracted from {video}")
    return rows


def _prompt(question: dict[str, Any]) -> str:
    options = "\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(question["options"]))
    return (
        "Answer the video multiple-choice question. Carefully inspect all frames. "
        "Return exactly one option letter and nothing else.\n\n"
        f"Question: {question['question']}\nOptions:\n{options}"
    )


def _parse_letter(text: str, option_count: int) -> int:
    match = re.search(r"\b([A-E])\b", text.upper())
    if not match:
        raise ValueError(f"model did not return an option letter: {text!r}")
    answer_id = ord(match.group(1)) - ord("A")
    if answer_id >= option_count:
        raise ValueError(f"model selected unavailable option: {text!r}")
    return answer_id


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def answer_one(record_key: str, question_index: int, frame_count: int = 24) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    volume.reload()
    db = json.loads(_annotation_path().read_text())
    record = db[record_key]
    question = record["mc_question"][question_index]
    source_id = record["metadata"]["video_id"]
    video = _find_video(record_key, source_id)
    route = classify(question)
    with tempfile.TemporaryDirectory(prefix="mcqa-frames-") as temp:
        sampled = _sample_frames(video, frame_count if route.dense else max(8, frame_count // 2), Path(temp))
        content = [{"type": "image", "image": Image.open(path).convert("RGB")} for _, path in sampled]
        content.append({"type": "text", "text": _prompt(question)})
        messages = [{"role": "user", "content": content}]
        processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=ROOT / "models" / "huggingface")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto", cache_dir=ROOT / "models" / "huggingface").eval()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        generated = model.generate(**inputs, max_new_tokens=8)
        answer_text = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    answer_id = _parse_letter(answer_text, len(question["options"]))
    result = {
        "record_key": record_key, "question_index": question_index,
        "answer": make_answer(question, answer_id), "model": MODEL_NAME,
        "route": route.__dict__, "confidence": None,
        "evidence_timestamps": [timestamp for timestamp, _ in sampled],
        "raw_output": answer_text,
    }
    output = ROOT / "predictions" / "validation" / f"{record_key}_{question_index}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    return result


@app.function(image=image, volumes={"/data": volume}, timeout=300)
def build_validation_submission() -> dict[str, Any]:
    volume.reload()
    db = json.loads(_annotation_path().read_text())
    predictions: dict[str, list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    for path in sorted((ROOT / "predictions" / "validation").glob("video_*.json")):
        result = json.loads(path.read_text())
        predictions.setdefault(result["record_key"], []).append(result["answer"])
        evidence.append(result)
    for answers in predictions.values():
        answers.sort(key=lambda item: int(item["id"]))
    errors = validate_predictions(predictions, db)
    output = ROOT / "predictions" / "validation" / "qwen_submission.json"
    evidence_path = ROOT / "predictions" / "validation" / "qwen_evidence.json"
    output.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    volume.commit()
    return {"questions": sum(len(items) for items in predictions.values()), "errors": errors, "output": str(output), "evidence": str(evidence_path)}


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=24 * 60 * 60)
def answer_validation(frame_count: int = 24, limit: int = 0) -> dict[str, Any]:
    """Run all validation questions with one cached model instance.

    `limit` is a smoke-test bound; zero means the complete validation split.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    volume.reload()
    db = json.loads(_annotation_path().read_text())
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto", cache_dir=ROOT / "models" / "huggingface"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=ROOT / "models" / "huggingface")
    submission: dict[str, list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    processed = 0
    for record_key, record in db.items():
        questions = record.get("mc_question") or []
        uncached = [
            (index, question)
            for index, question in enumerate(questions)
            if not (ROOT / "predictions" / "validation" / f"{record_key}_{index}.json").exists()
        ]
        if not uncached:
            for index, _ in enumerate(questions):
                cached = json.loads((ROOT / "predictions" / "validation" / f"{record_key}_{index}.json").read_text())
                submission.setdefault(record_key, []).append(cached["answer"])
                evidence.append(cached)
                processed += 1
            if limit and processed >= limit:
                break
            continue
        video = _find_video(record_key, str(record["metadata"].get("video_id", "")))
        with tempfile.TemporaryDirectory(prefix="mcqa-record-") as temp:
            sampled = _sample_frames(video, frame_count, Path(temp))
            images = [Image.open(path).convert("RGB") for _, path in sampled]
            for index, question in enumerate(questions):
                output_path = ROOT / "predictions" / "validation" / f"{record_key}_{index}.json"
                if output_path.exists():
                    cached = json.loads(output_path.read_text())
                    submission.setdefault(record_key, []).append(cached["answer"])
                    evidence.append(cached)
                    processed += 1
                    continue
                route = classify(question)
                content = [{"type": "image", "image": image} for image in images]
                content.append({"type": "text", "text": _prompt(question)})
                messages = [{"role": "user", "content": content}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
                generated = model.generate(**inputs, max_new_tokens=8)
                raw_output = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                answer_id = _parse_letter(raw_output, len(question["options"]))
                result = {
                    "record_key": record_key, "question_index": index,
                    "answer": make_answer(question, answer_id), "model": MODEL_NAME,
                    "route": route.__dict__, "confidence": None,
                    "evidence_timestamps": [timestamp for timestamp, _ in sampled],
                    "raw_output": raw_output,
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                submission.setdefault(record_key, []).append(result["answer"])
                evidence.append(result)
                processed += 1
                if limit and processed >= limit:
                    break
            if limit and processed >= limit:
                break
        if processed and processed % 8 == 0:
            volume.commit()
    output = ROOT / "predictions" / "validation" / "qwen_submission.json"
    evidence_path = ROOT / "predictions" / "validation" / "qwen_evidence.json"
    output.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    volume.commit()
    return {"processed": processed, "submission": str(output), "evidence": str(evidence_path), "model": MODEL_NAME}
