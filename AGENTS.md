# Challenge 2702 — Unified Multiple-Choice VideoQA

Read this file before making changes in this directory.

## Official links

- EvalAI: https://eval.ai/web/challenges/challenge-page/2702/overview
- Workshop: https://perception-test-challenge.github.io/
- Dataset and tools: https://github.com/google-deepmind/perception_test
- Perception Test paper: https://arxiv.org/abs/2305.13786

## Schedule

- Validation phase: June 5–August 27, 2026, 23:59:59 UTC.
- Test phase: July 15–August 27, 2026, 23:59:59 UTC.
- Technical report deadline: August 31, 2026, 23:59:59 UTC.
- Winner announcement: September 3, 2026.

## Task

Given a video with audio, a question, and three or five candidate answers, return exactly one correct choice. The unified track combines the original multiple-choice VideoQA questions with new questions that express lower-level perception tasks through multiple choice.

Original questions cover:

- Memory and event recall.
- Abstraction, matching, and counting.
- Physics, motion, and collisions.
- Semantics, actions, objects, and sounds.
- Descriptive, explanatory, predictive, and counterfactual reasoning.

New five-choice questions cover:

- Point tracking: identify the final marker corresponding to an initially highlighted surface point.
- Temporal localization: identify the colored time segment containing an action.
- Action-object interaction: identify the objects participating in a highlighted action.

## Data

Videos are high-resolution RGB with audio and last up to about 35 seconds.

| Split | Videos | Questions |
| --- | ---: | ---: |
| Validation | 189 | 454 |
| Test, original | 3,525 | 11,528 |
| Test, additional | 1,842 | 1,842 |

A valid test submission must answer both the original and additional sets. No training examples are provided for the new questions. If the existing Perception Test training split is used for the original questions, declare the submission as fine-tuned.

## Evaluation and submission

- Primary metric: top-1 accuracy.
- The leaderboard separately reports overall, original-set, and additional-set accuracy.
- Preserve the exact required question IDs, answer indexing, schema, and archive layout.
- Validate that every required question has exactly one answer before submission.

## Recommended implementation

1. Parse the question before decoding frames.
2. Classify its likely skill: audio, counting, temporal order, point tracking, action localization, object interaction, or general semantics.
3. Use uniform frame sampling for general questions and denser/event-directed sampling for motion and localization.
4. Extract audio and use ASR/audio-event features when relevant.
5. Ask a video-language model to score the supplied choices rather than generate an unconstrained answer.
6. Record per-question confidence, sampled timestamps, model version, and prompt version for reproducibility.

The first local baseline should use a quantized 7B-class video/vision-language model. Modal can provide an A100 80GB, permitting larger models or higher frame counts, but benchmark cost and throughput before processing the full test set.

## Rules

- Do not manually label test videos.
- Process each test video/question pair independently; do not exploit correlations among test questions.
- Do not use validation or test data for training, self-supervised learning, or prompting.
- Other datasets and the permitted Perception Test training split may be used as specified by the organizers.
- Public VLM APIs are allowed, but log model/version details and never commit credentials.
- A public test submission and technical report are required for prize eligibility.

