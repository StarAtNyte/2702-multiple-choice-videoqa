"""Label-free Molmo/Qwen3 answer router for cached per-question artifacts."""
from __future__ import annotations
import json
from pathlib import Path

# Chosen from capability differences, not individual question labels:
# Qwen3 is especially strong on fine-grained matching/tracking/recognition.
QWEN3_TAGS = {
    'feature matching', 'general knowledge', 'object recognition',
    'object_action_interaction', 'point_tracking', 'pretend_action',
    'solidity', 'state recognition',
}

def load_answers(root: Path, pattern: str) -> dict[tuple[str, int], dict]:
    out = {}
    for path in root.rglob(pattern):
        try:
            row = json.loads(path.read_text())
            out[(row['record_key'], int(row['question_index']))] = row
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return out

def route(q: dict, molmo: dict, qwen3: dict) -> dict:
    tag = (q.get('tag') or [''])[0]
    chosen = qwen3 if tag in QWEN3_TAGS else molmo
    return chosen

def build(annotation_path: Path, prediction_root: Path) -> tuple[dict, dict]:
    db = json.loads(annotation_path.read_text())
    molmo = load_answers(prediction_root, 'molmo_*.json')
    qwen3 = load_answers(prediction_root, 'qwen3_validation_*.json')
    submission, evidence = {}, {}
    for key, record in db.items():
        answers = []
        for i, q in enumerate(record.get('mc_question', [])):
            k = (key, i)
            chosen = route(q, molmo[k], qwen3[k])
            answers.append(chosen['answer'])
            evidence[f'{key}:{i}'] = {
                'model': chosen.get('model'), 'tag': (q.get('tag') or [''])[0],
                'molmo_answer': molmo[k]['answer']['answer_id'],
                'qwen3_answer': qwen3[k]['answer']['answer_id'],
            }
        submission[key] = answers
    return submission, evidence
