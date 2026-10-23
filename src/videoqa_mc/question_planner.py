"""Deterministic, label-free routing hints for Unified VideoQA questions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Route:
    skill: str
    frame_stride: int
    audio: bool
    dense: bool


def classify(question: dict[str, Any]) -> Route:
    text = f"{question.get('question', '')} {' '.join(question.get('tag') or [])}".lower()
    if any(k in text for k in ("sound", "hear", "audio", "voice", "say", "music")):
        return Route("audio", 15, True, True)
    if any(k in text for k in ("segment", "temporal", "when", "before", "after", "first", "last")):
        return Route("temporal", 5, False, True)
    if any(k in text for k in ("point", "marker", "highlighted surface")):
        return Route("point_tracking", 5, False, True)
    if any(k in text for k in ("count", "number of", "how many", "twice", "equal")):
        return Route("counting", 5, False, True)
    if any(k in text for k in ("involved", "objects", "interaction", "put", "place", "close")):
        return Route("interaction", 10, False, True)
    return Route("general", 15, False, False)
