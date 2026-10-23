# Challenge 2702 research log

## Official evidence

- EvalAI: https://eval.ai/web/challenges/challenge-page/2702/overview
- Leaderboard: https://eval.ai/web/challenges/challenge-page/2702/leaderboard
- Dataset/tools: https://github.com/google-deepmind/perception_test
- Unified validation annotations: `mc_vqa_unified_valid.zip`
- Unified validation videos: `annotated_vqa_valid.zip`

The EvalAI API reports validation phase `5341` and test phase `5342`. Both use
Top-1 metrics; the leaderboard schema separates overall, original, and extra
sets. The public leaderboard is JavaScript-rendered, so ranked rows must be
recorded from an actual public submission rather than inferred from search
snippets.

## Dataset inspection

The official unified validation annotation archive contains one JSON file with
189 records and 454 questions. Records retain the organizer's outer key and a
`metadata.video_id`; these are not always identical because multiple question
records can refer to one source video. Submission serialization must therefore
preserve the organizer record keys and question IDs exactly.

The unified task combines original 3-choice MC-VQA with additional 5-choice
questions for object-action interaction, temporal/action localization, and
point/perception tasks. The extra set has no permitted challenge examples for
fine-tuning; specialists must be zero-shot or trained only on eligible external
or official training data.

## Initial architecture hypothesis

1. Question parser and skill router.
2. Uniform low-rate frame index for every question, with dense/event-directed
   sampling for temporal, counting, and motion questions.
3. Timestamped Whisper/CLAP evidence for audio questions.
4. TimeLens2/Molmo2-style temporal and point evidence for localization tasks.
5. A video-language model (initially Qwen2.5-VL/Qwen3-VL or MOSS family) that
   scores the supplied choices, never free-generates an unconstrained answer.
6. Canonical option-permutation calibration to prevent option-zero collapse.
7. Evidence-card adjudication with per-question model/config/timestamp logs.

This combines the strongest reusable components from sibling projects 2705
(temporal grounding), 2706 (audio retrieval and constrained MCQ scoring), and
2707 (visual indexing) while keeping each question independent.

## Candidate benchmark evidence

- Frequency baseline trained from permitted MC-VQA training annotations: **242/454
  (0.5330)** on the unified validation set. This is a diagnostic only.
- Qwen2.5-VL-7B, uniform 16-frame prompts with exact-letter parsing: **219/384
  (0.5703)** on the currently completed cached validation subset. The run is
  still being completed; this is not a frozen score.
- Molmo2-8B successfully loaded and answered validation questions through the
  video-native API: **25/32 (0.7813)** on its initial smoke subset. This is not
  yet a full-split result, but it is materially stronger than Qwen on the same
  diagnostic subset and makes Molmo2 the primary candidate pending completion.

The completed Molmo2-8B validation run scored **313/454 = 0.6894** (100% coverage), below the internal 0.70 target. The corresponding public EvalAI test submission was nevertheless run as a provisional baseline after complete schema validation. EvalAI submission **579684** finished successfully with **overall Top-1 0.726552**, **original-set 0.778713**, and **extra-set 0.400109**. It used public test mode, 0-shot evaluation metadata, and method `Molmo2-8B direct video`. The test artifact contained all **13,370/13,370** questions with zero local validation errors. A second public specialist-prompt submission **579708** finished with **overall 0.728646**, **original 0.778713**, and **extra 0.415309**, slightly improving on submission 579684 (0.726552 overall; 0.778713 original; 0.400109 extra). Submission 579708 is therefore the current best.

Option permutation calibration and specialist routing remain candidates for improvement; the public score is a valid baseline, not evidence that the validation threshold was met.

## High-ROI architecture pivot

The current Molmo-only route is bottlenecked by the additional set (**41.53%** on the second public run). A prompt-only change is insufficient for an 85% target. The next candidate should be a routed hybrid:

1. **Point tracking:** decode sparse early/late frames, detect the flashed green marker, propagate its surface point with CoTracker or SAM2 video memory, then match against final candidate circles. SAM2 provides promptable video mask propagation with streaming memory; CoTracker provides learned point trajectories. This replaces unconstrained VLM guessing.
2. **Temporal localization:** detect the colored bar/segment overlay in the top image strip using HSV/color masks across dense frames, align each candidate color to the highlighted action interval, and ask Molmo only to identify the action if needed.
3. **Object-action interaction:** use dense frames plus an object-centric prompt; optionally add SAM2 masks/connected components before Molmo adjudication. Route by question text and option count, never by answer labels.
4. **Original 3-choice:** retain Molmo direct-video as the primary scorer, then add a stronger independent scorer (Qwen-VL or an allowed hosted VLM) only where confidence is low. Calibrate routing on the permitted train/dev fold, not validation/test labels.

The official SAM2 and CoTracker implementations were inspected as candidate specialist dependencies. This architectural split directly targets the public leaderboard gap: our original-set score is already 77.87%, while the extra set is 41.53%; improving the extra set is the dominant path toward 85%.

A Hugging Face review also identified **Qwen3-VL-8B-Instruct** as a high-ROI local ablation. Its model card advertises native video understanding, 256K context, temporal/spatial position encoding, DeepStack visual features, and text–timestamp alignment—capabilities directly relevant to action order and marker localization. A Modal Qwen3 smoke job is queued but has not obtained a GPU task yet. (Later run: the full 454-question validation job completed on A10G with 12 frames/video.)

The Qwen3 validation run scored **294/454 = 64.76%** standalone versus the cached Molmo validation predictions at **335/454 = 73.79%**. Despite that lower standalone score, their complementary oracle reached **387/454 = 85.24%** (at least one model correct). A simple tag-level router reached **357/454 = 78.63%**, confirming that confidence/disagreement routing—not blind replacement—is the promising next experiment. Implemented this reproducible label-free capability router in `src/videoqa_mc/ensemble_router.py`; it selects Qwen3 for matching/recognition/tracking-style tags and Molmo elsewhere. It is intentionally not yet applied to test until a stronger confidence signal is added. The Qwen3 first-token A–E margin experiment is running on `whitestjohn0`; on the currently available validation artifacts, naive confidence replacement is worse than Molmo (about 70% versus 73.79%), so confidence alone is not yet a valid deployment router.

A direct review of the Perception Test 2025 unified-VQA report provides a stronger architecture signal than another local prompt ablation: its best zero-shot submission used a **dual-model Seed 1.6 Vision + Qwen-VL-Max fallback**, reaching 83.7% original and 65.4% novel-set accuracy. Its best fine-tuned submission combined **Qwen-2.5-VL with GLM-4.5V and Seed-1.6-Vision APIs**. The report also notes a SAM2 + AllTracker pipeline as the leading approach for the separate unified point/object tracking task. Therefore the highest-ROI path to 85% is a confidence-routed multi-model ensemble (hosted strong VLM plus local Molmo), with deterministic tracking/bar specialists as fallback—not Molmo-only inference. A first CPU optical-flow point-tracking prototype was tested on all 18 point-tracking validation questions and scored only **6/18 (33.3%)**, so it is not a replacement for a learned tracker yet; the specialist lane was stopped rather than applied blindly.

## Grounding/VQA technique research (2026-08-24)

Two concrete bugs found and fixed, not model-size limitations:

1. **32B FP8 vLLM job silently died at 32/454.** `src/videoqa_mc/jobs/modal_qwen3_32b_vllm.py` had no
   try/except around per-question generation/parsing; one unparseable answer
   (`max_tokens=8` truncation) raised and killed the whole 12h H100 job. The
   56.25% figure quoted for it was noise from an incomplete n=32 run, not
   comparable to the full-454 8B/Molmo baselines. Fixed: per-question
   try/except writing `*.error.json` records instead of crashing, stdout
   logging of every ok/error/skip, `max_tokens` raised to 128 with an explicit
   `Answer: <letter>` anchor in the prompt (see technique #2 below), and the
   confound that it also switched to native-video decoding (see below) is
   still unresolved — a rerun should still be compared against an 8B run using
   the *same* native-video path before concluding anything about model size.
2. **CoTracker3 point-tracking specialist scored 6.7%, below the 20% chance
   floor for 5-choice questions** (`src/videoqa_mc/jobs/modal_cotracker.py`, and the OpenCV
   optical-flow variant in `src/videoqa_mc/jobs/modal_specialists.py` scored 0%/33%). Root cause:
   `load_frames` uniformly resampled every video down to a fixed 64 frames
   regardless of source length/fps, leaving multi-frame motion gaps the
   tracker could not bridge. Official CoTracker3 usage
   ([facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker),
   query format confirmed as `(t, x, y)` per-point, which our code already had
   right) runs on native-density video. Fixed `load_frames` to read
   sequentially at native density (was also using `cap.set(POS_FRAMES)`
   keyframe-seeking, which is not exact-frame-accurate on compressed video;
   sequential read fixes that too). Re-run on H200 at full native density
   (uncapped for clips up to 35s): still scored 1/15 (6.7%), unchanged from
   the broken 64-frame run and still at/below the 20% chance floor for
   5-choice. Manual inspection of all 15 predictions found no index-collapse
   pattern, so the remaining failure is in point localization/tracking
   fidelity itself (four independent from-scratch attempts have now landed at
   or below chance except one early prototype at 33%) — logged as rejected
   again in `experiments/20260824_cotracker3_native_h200.json`. Do not invest
   further Modal time in a from-scratch tracker here without adopting the
   actual SAM2+AllTracker reference pipeline.
3. **Case-sensitivity bug silently corrupted ≥12.6% of accepted 32B FP8
   answers.** After fixing bug #1 above, the CoT-then-anchor prompt
   (`...Answer: <letter>`) was added along with `letter()`'s anchor regex
   `re.search(r'Answer:\s*([A-E])\b', s.upper())` — but this searches a
   mixed-case literal against an *uppercased* string, so `'Answer:'` never
   matches `'ANSWER:'` and the anchor silently never fires. Every accepted
   answer fell through to the unanchored fallback (first standalone A-E
   letter anywhere in the raw text), which happily grabs an incidental early
   letter mention (spelled-out words, other options discussed mid-reasoning)
   instead of the model's actual final answer. Audited all 437
   then-accepted predictions: 79 contained a literal "answer:" (i.e. the
   model did follow the CoT format), and 55 of those 79 (70%) had a
   different, verifiably-wrong letter recorded than what the model actually
   concluded. Fixed by uppercasing the anchor literal too
   (`ANSWER:\s*([A-E])\b` against `s.upper()`). The entire corrupted
   `qwen3_32b_fp8_fps05_*` prediction set (454 files) was purged from the
   volume and the full validation run redone from scratch rather than
   patched, since the corruption rate among answers without a literal
   "answer:" in them was unknown and unverifiable after the fact. Lesson:
   when uppercasing one side of a regex match for case-insensitivity, either
   uppercase both sides or use `re.IGNORECASE` — never mix.
4. **After fixing bug #3, the rerun scored 40.4% — worse than every prior
   32B attempt, including the original broken 64-frame/8-token run (53.1%).**
   Root cause: `max_tokens=128` was too small for `Answer: <letter>` CoT to
   land (only 97/453 = 21% of answers ever reached the anchor before
   truncating), so 79% fell through to the unanchored fallback regex
   `\b([A-E])\b` scanned over the entire uppercased free-text reasoning.
   That fallback doesn't just risk grabbing an incidental option-letter
   mention (bug #3's framing) — it matches the English article "a" in
   almost every sentence once uppercased, since "a" is itself a standalone
   single-letter word. Audit: fallback-bucket accuracy was 30.6% (356
   answers) vs 76.3% for anchored answers (97 answers), and 99.7% of
   fallback answers were index 0 (answer_id==0) regardless of the true
   answer — the fallback was measuring nothing, just always picking A.
   Fixed by removing the unanchored fallback entirely (letter() now
   requires the ANSWER: anchor or raises) and raising max_tokens to 400
   (the value retry_failed already proved gets ~95% of answers to a
   concluding anchor). Purged and reran clean again. Lesson: a "fallback"
   that silently succeeds on ordinary English text is worse than no
   fallback — it hides a systematic bias (always A) inside what looks like
   normal accuracy, and only a per-bucket accuracy breakdown (not just an
   aggregate mismatch count) revealed it.

Research pulled from the official challenge organizers and the actual
Perception Test MC-VQA challenge winners:

- **Perception Test 2025 organizer summary**
  ([arXiv:2601.06287](https://arxiv.org/html/2601.06287v1)): best zero-shot
  unified result was 81.2% (Seed 1.6 Vision + Qwen-VL-Max, with fallback: "if
  the first model failed to generate an answer, the second model would be
  called"). Original 3-choice zero-shot: 83.7%; new 5-choice: 65.4% (an 18.3pt
  drop vs 3-choice — matches our own 77.87%/41.53% split, so the difficulty
  gap is expected, not a sign our pipeline is unusually broken there). Top
  tracking-task teams used SAM2 + AllTracker pipelines. Physics and
  Abstraction were the hardest original-set categories.
- **First-place Second Perception Test MC-VQA solution**
  ([arXiv:2409.13538](https://arxiv.org/html/2409.13538v1), 76.47% top-1):
  fine-tuned QwenVL2-7B (LoRA rank 8-16, 5-fold CV — not directly reusable
  zero-shot, would need declaring as fine-tuned and only applies to the
  original set since no training data exists for the new 5-choice questions).
  Three techniques ARE directly reusable zero-shot:
  - **Option-permutation TTA + majority vote**: re-infer each question with
    2-3 shuffled option orders, vote. This is exactly the "canonical
    option-permutation calibration to prevent option-zero collapse" already
    flagged as a pending candidate above — never implemented. Cheapest,
    highest-confidence next experiment; needs no new model or training.
  - **Frame budget matters a lot more than we've been using**: their winning
    config used up to 60 frames at 560×630 (352,800px); we run 12-24 frames at
    ≤512×512 (our vLLM 32B run used max_pixels 512×512, i.e. ≤262,144px, and
    the native-8B path uses only 224×224). Under-sampling could be
    suppressing accuracy independent of model choice.
  - **CoT room + explicit answer anchor**: their prompt used "think step by
    step" style prompting; our scripts capped `max_tokens=8` with zero room to
    reason, forcing an immediate raw letter guess. Now changed in
    `src/videoqa_mc/jobs/modal_qwen3_32b_vllm.py` to allow 128 tokens ending in `Answer: <letter>`,
    parsed by anchor-first regex. Not yet validated on a full run.

Not yet implemented, deferred pending direction: temporal-localization
color-bar specialist (sketched in the architecture pivot above, no code
exists yet) and option-permutation TTA wired into the Molmo/Qwen3 8B
pipelines (cheap, but is new experiment surface, not a bugfix — confirm
priority before spending more Modal GPU time).

## Data policy

Raw media, extracted frames/audio, model caches, embeddings, predictions, and
credentials belong on the Modal Volume `perception-test-2026-data`; they must
not enter this repository. Validation answers are for one external evaluation
only and cannot be used for tuning or model selection.
