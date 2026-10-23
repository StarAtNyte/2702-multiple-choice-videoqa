"""Local evaluation for organizer-format MCQA artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate(predictions: dict[str, Any], annotations: dict[str, Any]) -> dict[str, Any]:
    total = correct = 0
    by_tag: dict[str, list[int]] = {}
    for record_key, record in annotations.items():
        answers = {int(item["id"]): item for item in predictions.get(record_key, [])}
        for question in record.get("mc_question") or []:
            total += 1
            answer = answers.get(int(question["id"]))
            hit = int(answer is not None and answer.get("answer_id") == question.get("answer_id"))
            correct += hit
            for tag in question.get("tag") or ["untagged"]:
                by_tag.setdefault(tag, []).append(hit)
    return {
        "correct": correct,
        "total": total,
        "top1": correct / total if total else 0.0,
        "by_tag": {tag: sum(values) / len(values) for tag, values in sorted(by_tag.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.predictions.read_text()), json.loads(args.annotations.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
