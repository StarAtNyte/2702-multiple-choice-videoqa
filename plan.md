# Challenge 2702 execution plan

1. Ingest official annotations and video archives into Modal Volume `perception-test-2026-data` under profile `baptinjean`.
2. Evaluate local Molmo/Qwen3 baselines and the deterministic label-free router on validation; do not use validation labels for routing.
3. Freeze the strongest permitted complete candidate, validate combined original + additional test coverage, and preserve a fallback.
4. Submit the complete artifact to EvalAI Test Phase 5342 and record the receipt.

All raw media, model caches, and predictions remain outside git on the Modal volume.
