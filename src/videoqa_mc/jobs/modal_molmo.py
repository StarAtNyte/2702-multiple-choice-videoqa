"""Molmo2-8B ablation lane for video-native QA and tracking questions."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import modal

from videoqa_mc.mcqa_core import make_answer

ROOT = Path("/data/2702-multiple-choice-videoqa")
MODEL_NAME = os.getenv("MCQA_MOLMO_MODEL", "allenai/Molmo2-8B")
app = modal.App("unified-mcqa-2702-molmo")
volume = modal.Volume.from_name("perception-test-2026-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "ca-certificates")
    .pip_install("accelerate>=1.9,<2", "decord2", "einops", "molmo-utils", "pillow>=11,<13", "torch==2.7.1", "torchvision==0.22.1", "transformers==4.57.1")
    .env({"HF_HOME": str(ROOT / "models" / "huggingface")})
    .add_local_python_source('videoqa_mc')
)


def _video(record_key: str, source_id: str) -> Path:
    roots = [ROOT / "raw" / "validation", ROOT / "raw" / "full_validation", ROOT / "raw" / "test", ROOT / "raw" / "test_unified"]
    for identifier in (record_key, source_id):
        for root in roots:
            matches = list(root.rglob(f"{identifier}*.mp4"))
            if matches:
                return matches[0]
    raise FileNotFoundError(record_key)


def _frames(video: Path, directory: Path, count: int = 8) -> list[tuple[float, Path]]:
    import subprocess
    directory.mkdir(parents=True, exist_ok=True)
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)], text=True).strip())
    rows = []
    for i in range(count):
        timestamp = duration * i / max(1, count - 1)
        path = directory / f"{i:03d}.jpg"
        safe = min(timestamp, max(0, duration - 0.05))
        if not path.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{safe:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=768:-2", "-q:v", "3", str(path)], check=True)
        if path.exists():
            rows.append((timestamp, path))
    return rows


def _letter(text: str, count: int) -> int:
    match = re.search(r"\b([A-E])\b", text.upper())
    if not match:
        raise ValueError(text)
    index = ord(match.group(1)) - 65
    if index >= count:
        raise ValueError(text)
    return index


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=24 * 60 * 60)
def answer_validation_images(frame_count: int = 8, limit: int = 0) -> dict[str, Any]:
    import tempfile
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    volume.reload()
    files = sorted((ROOT / "manifests" / "validation_annotations").glob("*.json"))
    db = json.loads(files[0].read_text())
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto").eval()
    processed = 0
    for record_key, record in db.items():
        video = _video(record_key, str(record["metadata"].get("video_id", "")))
        with tempfile.TemporaryDirectory(prefix="molmo-frames-") as temp:
            sampled = _frames(video, Path(temp), frame_count)
            images = [Image.open(path).convert("RGB") for _, path in sampled]
            for index, question in enumerate(record.get("mc_question") or []):
                destination = ROOT / "predictions" / "validation" / f"molmo_fast_{record_key}_{index}.json"
                if destination.exists():
                    processed += 1
                    continue
                options = "\\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(question["options"]))
                prompt = f"Inspect all video frames and answer with exactly one option letter.\\nQuestion: {question['question']}\\nOptions:\\n{options}"
                content = [{"type": "text", "text": prompt}] + [{"type": "image", "image": image} for image in images]
                messages = [{"role": "user", "content": content}]
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
                inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=16)
                output = processor.tokenizer.decode(generated[0, inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
                answer_id = _letter(output, len(question["options"]))
                result = {"record_key": record_key, "question_index": index, "answer": make_answer(question, answer_id), "model": MODEL_NAME, "raw_output": output, "frame_timestamps": [t for t, _ in sampled]}
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
                processed += 1
                if limit and processed >= limit:
                    break
            if limit and processed >= limit:
                break
        if processed and processed % 8 == 0:
            volume.commit()
    volume.commit()
    return {"processed": processed, "model": MODEL_NAME, "mode": "multi-image"}


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=24 * 60 * 60)
def answer_validation(max_fps: int = 4, limit: int = 0, record_start: int = 0, record_end: int = 0, prefix: str = "molmo", specialist: bool = False) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from molmo_utils import process_vision_info

    volume.reload()
    files = sorted((ROOT / "manifests" / "validation_annotations").glob("*.json"))
    db = json.loads(files[0].read_text())
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto").eval()
    processed = 0
    records = list(db.items())[record_start:record_end or None]
    for record_key, record in records:
        video = _video(record_key, str(record["metadata"].get("video_id", "")))
        for index, question in enumerate(record.get("mc_question") or []):
            destination = ROOT / "predictions" / "validation" / f"{prefix}_{record_key}_{index}.json"
            if destination.exists():
                processed += 1
                continue
            options = "\\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(question["options"]))
            if specialist and len(question["options"]) == 5:
                tag = " ".join(question.get("tag") or []).lower()
                if "point_tracking" in tag:
                    instruction = "Track the same physical surface point from the brief initial colored flash to the final candidate circles, accounting for object motion and occlusion."
                elif "action_order" in tag:
                    instruction = "Inspect the complete video and reconstruct the chronological action sequence before comparing each option."
                elif "pretend_action" in tag:
                    instruction = "Use the colored top bar to locate the requested segment, then judge whether the described action was incorrect or pretend."
                else:
                    instruction = "Use the colored top bar to isolate the highlighted interval, then identify only the objects participating in that action."
                prompt = f"{instruction} Do not guess from option order. Answer with exactly one option letter.\\nQuestion: {question['question']}\\nOptions:\\n{options}"
            else:
                prompt = f"Inspect the video and answer with exactly one option letter.\\nQuestion: {question['question']}\\nOptions:\\n{options}"
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "video", "video": str(video), "max_fps": max_fps}]}]
            _, videos, video_kwargs = process_vision_info(messages)
            videos, metadata = zip(*videos)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(videos=list(videos), video_metadata=list(metadata), text=text, padding=True, return_tensors="pt", **video_kwargs)
            inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16)
            output = processor.tokenizer.decode(generated[0, inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            answer_id = _letter(output, len(question["options"]))
            result = {"record_key": record_key, "question_index": index, "answer": make_answer(question, answer_id), "model": MODEL_NAME, "raw_output": output, "max_fps": max_fps}
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
            processed += 1
            if limit and processed >= limit:
                break
        if limit and processed >= limit:
            break
        if processed and processed % 8 == 0:
            volume.commit()
    volume.commit()
    return {"processed": processed, "model": MODEL_NAME}


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=3600)
def answer_one(record_key: str, question_index: int, max_fps: int = 4) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from molmo_utils import process_vision_info

    volume.reload()
    files = sorted((ROOT / "manifests" / "validation_annotations").glob("*.json"))
    db = json.loads(files[0].read_text())
    record = db[record_key]
    question = record["mc_question"][question_index]
    video = _video(record_key, str(record["metadata"].get("video_id", "")))
    options = "\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(question["options"]))
    prompt = f"Inspect the video and answer with exactly one option letter.\nQuestion: {question['question']}\nOptions:\n{options}"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "video", "video": str(video), "max_fps": max_fps}]}]
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto").eval()
    _, videos, video_kwargs = process_vision_info(messages)
    videos, metadata = zip(*videos)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(videos=list(videos), video_metadata=list(metadata), text=text, padding=True, return_tensors="pt", **video_kwargs)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16)
    output = processor.tokenizer.decode(generated[0, inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
    answer_id = _letter(output, len(question["options"]))
    result = {"record_key": record_key, "question_index": question_index, "answer": make_answer(question, answer_id), "model": MODEL_NAME, "raw_output": output, "max_fps": max_fps}
    destination = ROOT / "predictions" / "validation" / f"molmo_{record_key}_{question_index}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    return result


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=24 * 60 * 60)
def answer_test(record_start: int = 0, record_end: int = 0, max_fps: int = 1, unified: bool = False, specialist: bool = False) -> dict[str, Any]:
    """Run label-free Molmo inference over an original or additional test shard."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from molmo_utils import process_vision_info

    volume.reload()
    if unified:
        annotation_path = ROOT / "manifests" / "mc_vqa_unified_test.json"
        output_dir = ROOT / "predictions" / ("test_unified_special" if specialist else "test_unified")
    else:
        annotation_path = sorted((ROOT / "manifests" / "test_annotations").glob("*.json"))[0]
        output_dir = ROOT / "predictions" / ("test_special" if specialist else "test")
    db = json.loads(annotation_path.read_text())
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, trust_remote_code=True, dtype="auto", device_map="auto").eval()
    records = list(db.items())[record_start:record_end or None]
    processed = 0
    for record_key, record in records:
        video = _video(record_key, str(record.get("metadata", {}).get("video_id", "")))
        for index, question in enumerate(record.get("mc_question") or []):
            destination = output_dir / f"molmo_{record_key}_{index}.json"
            if destination.exists():
                processed += 1
                continue
            options = "\\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(question["options"]))
            if specialist and len(question["options"]) == 5:
                prompt = f"Analyze this video frame-by-frame. Track the visual evidence requested: colored bars may mark temporal segments, and circles may mark corresponding surface points. For action questions, identify only objects visibly involved. Do not guess from option order. Answer with exactly one option letter.\\nQuestion: {question['question']}\\nOptions:\\n{options}"
            else:
                prompt = f"Inspect the video and answer with exactly one option letter.\\nQuestion: {question['question']}\\nOptions:\\n{options}"
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "video", "video": str(video), "max_fps": max_fps}]}]
            _, videos, video_kwargs = process_vision_info(messages)
            videos, metadata = zip(*videos)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(videos=list(videos), video_metadata=list(metadata), text=text, padding=True, return_tensors="pt", **video_kwargs)
            inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16)
            output = processor.tokenizer.decode(generated[0, inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
            answer_id = _letter(output, len(question["options"]))
            result = {"record_key": record_key, "question_index": index, "answer": make_answer(question, answer_id), "model": MODEL_NAME, "raw_output": output, "max_fps": max_fps, "split": "test_unified" if unified else "test"}
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
            processed += 1
        if processed and processed % 8 == 0:
            volume.commit()
    volume.commit()
    return {"processed": processed, "model": MODEL_NAME, "split": "test_unified" if unified else "test"}
