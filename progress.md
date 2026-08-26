# Challenge 2702 progress

- Profile: `baptinjean` (activated as the default Modal profile).
- Downloads: annotations complete; unified test, original test, and full validation archives complete. Full validation manifest reports 2,927 extracted videos and zero missing validation videos.
- Validation: Molmo-v2 335/454 (73.79%); Qwen3 294/454 (64.76%); label-free tag router 357/454 (78.63%), complete coverage.
- Final artifact: 4,988 videos and exactly 13,370 answers; combined schema/coverage validation returned zero errors.
- Public EvalAI Test Phase 5342: submission 579708, overall 72.8646%, original 77.8713%, additional 41.5309%.
- Fresh resubmission 580119 (2026-08-24, same validated artifact) reproduced identical scores: overall 72.8646%, original 77.8713%, additional 41.5309%. Local schema/coverage validation: 13,370/13,370, zero errors.
- Public EvalAI Test Phase 5342 submission 580630 (2026-08-26): overall 78.1227%, original 77.8713%, additional 79.6960%. It preserves all 11,528 original-set answers from submission 580119 and replaces all 1,842 additional-set answers with Gemini 3.7 Flash outputs. Local validation: 13,370/13,370, exact ID set, zero errors.
- Session 2026-08-24: fixed silent-crash bugs in `modal_qwen3_32b_vllm.py` and `modal_cotracker.py` (no error handling) and a case-sensitivity regex bug that had corrupted ≥12.6% of 32B FP8 answers; none of these ablations beat the Molmo baseline on validation, so the submitted method is unchanged.
- Tests: `python -m unittest discover -s tests -v` passes 5/5.
