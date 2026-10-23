"""Dependency-light contracts for Unified Multiple-Choice VideoQA."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_questions(db: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield stable question keys and question records from organizer JSON."""
    for video_id, video in db.items():
        for index, question in enumerate(video.get("mc_question") or []):
            qid = str(question.get("id", index))
            yield f"{video_id}:{qid}", question


def question_count(db: dict[str, Any]) -> int:
    return sum(1 for _ in iter_questions(db))


def validate_predictions(predictions: dict[str, Any], db: dict[str, Any]) -> list[str]:
    """Validate coverage and answer text without reading ground-truth answers."""
    errors: list[str] = []
    expected: dict[str, dict[str, Any]] = dict(iter_questions(db))
    seen: set[str] = set()

    if not isinstance(predictions, dict):
        return ["submission must be a JSON object"]

    for video_id, answers in predictions.items():
        if video_id not in db:
            errors.append(f"unexpected video id: {video_id}")
            continue
        if not isinstance(answers, list):
            errors.append(f"{video_id}: answers must be a list")
            continue
        for answer in answers:
            if not isinstance(answer, dict):
                errors.append(f"{video_id}: answer must be an object")
                continue
            qid = str(answer.get("id"))
            key = f"{video_id}:{qid}"
            if key not in expected:
                errors.append(f"unexpected question: {key}")
                continue
            if key in seen:
                errors.append(f"duplicate question: {key}")
            seen.add(key)
            options = expected[key].get("options") or []
            answer_id = answer.get("answer_id")
            if not isinstance(answer_id, int) or not 0 <= answer_id < len(options):
                errors.append(f"{key}: invalid answer_id {answer_id!r}")
            elif answer.get("answer") != options[answer_id]:
                errors.append(f"{key}: answer text does not match answer_id")

    missing = sorted(set(expected) - seen)
    errors.extend(f"missing question: {key}" for key in missing)
    return errors


def make_answer(question: dict[str, Any], answer_id: int) -> dict[str, Any]:
    options = question.get("options") or []
    if not isinstance(answer_id, int) or not 0 <= answer_id < len(options):
        raise ValueError("answer_id outside question options")
    return {"id": question.get("id", 0), "answer_id": answer_id, "answer": options[answer_id]}
