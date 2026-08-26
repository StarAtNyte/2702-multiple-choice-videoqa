# Challenge 2702: evidence-backed route to 84%+

**Audience:** project team  
**Date:** 2026-08-24  
**Scope:** repository audit, prior Perception Test winners, and a validation-first plan for the remaining submission window. No test labels were used and no new cloud inference was launched during this audit.

## Executive answer

The current best artifact is EvalAI submission **579708: 72.8646% overall, 77.8713% original, 41.5309% additional**. The limiting problem is the additional five-choice set, but raising that set alone cannot produce 84% overall. Because the split is 11,528 original and 1,842 additional questions, 84% overall requires combinations such as **88% original + 59% additional**, or **90% original + 46.5% additional**. The first combination is the credible target; it requires a substantial original-set ensemble gain and a roughly 17.5-point additional-set gain.

The recommended path is a *single unified submission with capability-routed evidence and multi-model scoring*: retain Molmo2 as the current local anchor, add a strong independent scorer (the prior winner used Seed 1.6 Vision with Qwen-VL-Max fallback), use deterministic visual evidence only as input to the VLM rather than as a separate answer interface, and deploy option-permutation test-time augmentation plus calibrated disagreement routing. Do not spend more time on the current CoTracker-from-scratch implementation: four attempts are at or below the 20% five-choice chance floor.

## What the repository proves

- Frequency baseline: 53.30% (242/454 validation).
- Molmo2-8B full validation: 68.94% (313/454); cached later Molmo predictions used by routing: 73.79% (335/454).
- Qwen3 validation: 64.76% (294/454), but complementary oracle with Molmo: 85.24% (387/454).
- Simple tag router: 78.63% (357/454); confidence-only replacement was worse than Molmo.
- Public 579708: 72.8646% overall, 77.8713% original, 41.5309% additional, complete 13,370-question coverage.
- CoTracker3 native-density rerun: 1/15 (6.7%) point questions; reject this specialist until the actual SAM2+AllTracker reference pipeline is adopted.
- Qwen3-32B FP8 experiments exposed parser failures: mixed-case anchor matching, insufficient generation budget, and an unsafe fallback that matched the article “a” and collapsed to option A. The parser must fail closed and every answer must be bucket-audited.

## Prior winners and what transfers

The First Place Solution to the Second Perception Test MC-VQA challenge reports **76.47% top-1**, using fine-tuned QwenVL2-7B, model ensembling, and test-time augmentation. Its zero-shot transferable ideas are option permutation TTA, more frames, and explicit reasoning/answer anchoring; fine-tuning is only allowable for the original task and must be declared.

The organizer’s 2025 challenge report is the closest published unified benchmark. Its strongest zero-shot unified system used **Seed 1.6 Vision + Qwen-VL-Max fallback**, reporting **83.7% original and 65.4% novel five-choice**. The report also identifies SAM2 + AllTracker-style pipelines among the strongest tracking systems. This is evidence for independent-model fallback and learned tracking, not proof that an API score transfers unchanged to this year’s split.

Sources: [2025 organizer report](https://arxiv.org/html/2601.06287v2#S4.SS1), [2024 first-place solution](https://arxiv.org/html/2409.13538v1), [official challenge site](https://perception-test-challenge.github.io/), [official dataset/tools](https://github.com/google-deepmind/perception_test).

## Highest-value execution plan

### 1. Freeze a leakage-safe local selection protocol

Create 5 folds over the permitted original training annotations only. Hold out one fold for model/router selection, never use validation/test answers for prompt design, and report original/additional separately. Every prediction record should contain model, checkpoint, prompt hash, frame timestamps/count, audio branch, parser result, confidence, and error state. Fail closed on malformed outputs.

### 2. Implement the cheap ablations first

1. **Option-permutation TTA:** run two or three deterministic permutations, invert each answer to canonical option IDs, and majority-vote. Apply to Molmo and Qwen3 on the local fold. This directly attacks option-position bias and costs 2–3× inference, but no new model or labels.
2. **Frame/resolution sweep:** compare 16, 32, and 60 frames at a fixed model, preserving native timestamps. The prior winner used up to 60 frames; our recent paths often used 12–24 and, for one native Qwen path, 224px inputs.
3. **Disagreement router:** train a tiny fold-only calibrator over model identity, option count, question tag, entropy/margin, output validity, and permutation agreement. Use Qwen3 only where its fold likelihood beats Molmo; otherwise retain Molmo. The 85.24% oracle shows the headroom, while the 78.63% hand router shows that tag routing alone is insufficient.
4. **Audio branch:** for speech/sound tags, append timestamped Whisper transcript or CLAP choice scores as evidence, never as an unconstrained answerer. Compare video-only versus video+audio on the same folds.

### 3. Target the additional set with evidence extraction

Use question text and five-option structure to detect point tracking, temporal localization, and object-action interaction. For point questions, use the official/reference SAM2+AllTracker implementation or a validated equivalent; do not ship the failed CoTracker prototype. For colored temporal bars, detect the overlay and produce an interval evidence card. For object interaction, produce dense frame contact/crop evidence and let the VLM score the supplied choices. These components should *rerank or condition the unified VLM*, preserving the challenge’s multiple-choice interface.

### 4. Add one independent high-end scorer if credentials and budget permit

Benchmark Seed 1.6 Vision, Qwen-VL-Max, or Gemini on the 454-question validation split with identical canonical options and fail-closed parsing. Cache video uploads and use it only as a disagreement/low-confidence fallback if its fold accuracy and calibration justify cost. The prior winner’s fallback pattern is more credible than blind averaging.

### 5. Gate the final submission

Do not submit a candidate unless it has complete 454/454 local coverage, zero schema errors, permutation/parser audit, and a frozen manifest. Then generate exactly 13,370 answers, validate IDs and indexing, and preserve the current 579708 artifact as rollback.

## Decision thresholds

Use the following validation gates:

| Gate | Required evidence |
|---|---|
| Parser safety | 100% parseable; no unanchored fallback; option-index histogram audited |
| Local model | complete 454/454; report original/additional and all major tags |
| Router | beats the best single model on every held-out fold or is not deployed |
| Additional specialist | >20% on held-out five-choice point subset before any test use |
| Submission | 13,370/13,370 IDs, zero duplicates/missing/errors |

## Risks and limits

An 84% overall score is not guaranteed: the current gap is large, and the 2025 novel-set result shows that five-choice perception questions remain intrinsically difficult. The 2025 API numbers are not directly comparable to this submission, and the live 2026 leaderboard is JavaScript-rendered; ranks should be verified from EvalAI immediately before claiming a win. The proposed target is therefore an experimental threshold with explicit go/no-go gates, not a forecast.

