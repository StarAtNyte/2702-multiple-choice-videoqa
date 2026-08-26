"""Deterministic video-disjoint folds for permitted training annotations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def split_records(db: dict[str, Any], dev_fraction: float = 0.1) -> tuple[dict[str, Any], dict[str, Any]]:
    train: dict[str, Any] = {}
    dev: dict[str, Any] = {}
    for key, record in db.items():
        source = str(record.get("metadata", {}).get("video_id", key))
        score = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (dev if score < dev_fraction else train)[key] = record
    return train, dev


def write_folds(annotation_path: str | Path, output_dir: str | Path, dev_fraction: float = 0.1) -> dict[str, int]:
    db = json.loads(Path(annotation_path).read_text())
    train, dev = split_records(db, dev_fraction)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (destination / "dev.json").write_text(json.dumps(dev), encoding="utf-8")
    return {"train_records": len(train), "dev_records": len(dev), "train_questions": sum(len(v.get("mc_question") or []) for v in train.values()), "dev_questions": sum(len(v.get("mc_question") or []) for v in dev.values())}
