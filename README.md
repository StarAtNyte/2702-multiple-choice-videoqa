# Unified Multiple-Choice VideoQA

Capability-routed video question answering for the Perception Test 2026 unified multiple-choice track.

Question classification → frame/audio evidence → model and specialist scoring → routed answers and schema validation.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --extra dev
uv run pytest
uv run modal run -m videoqa_mc.jobs.modal_data --help
```

The local test suite uses synthetic inputs. Model inference runs in Modal;
GPU frameworks and model-specific dependencies are defined in each job's image,
so local setup does not download model weights.

## Running the pipeline

Authenticate with Modal and configure the profile and any model/API secrets required
by the selected job. The commands below launch remote work and may incur cloud costs.

```bash
uv run modal run -m videoqa_mc.jobs.modal_data --action bootstrap
uv run modal run -m videoqa_mc.jobs.modal_baselines --help
```

Use module paths for every job: `uv run modal run -m videoqa_mc.jobs.<module>`.
Local tools also run as modules:

```bash
uv run python -m videoqa_mc.tools.evaluate --help
```

## Code layout

| Path | Purpose |
| --- | --- |
| [`src/videoqa_mc/`](src/videoqa_mc/) | Shared components: `mcqa_core.py`, `question_planner.py`, `folds.py`, and `ensemble_router.py` |
| [`src/videoqa_mc/jobs/`](src/videoqa_mc/jobs/) | Modal data jobs, inference pipelines, and experiments |
| [`src/videoqa_mc/tools/`](src/videoqa_mc/tools/) | Local evaluation and operational utilities |
| [`tests/`](tests/) | Synthetic regression tests |
| [`docs/`](docs/README.md) | Research, architecture, experiment history, and recorded results |

The previous root-level Python modules now live under `src/videoqa_mc/`.
Imports use the package namespace, and Modal images include that package for remote execution.

## Data and evaluation

Persistent storage: `perception-test-2026-data` (see the data job for track paths). Keep raw media, extracted frames, model weights,
credentials, and new prediction artifacts out of Git. Existing checked-in experiment
records are retained for traceability.

Read [`AGENTS.md`](AGENTS.md) for challenge-specific constraints. Preserve organizer
IDs, option order or track identities, and the submission schema. Validation and test
examples must not become training or few-shot prompt data. Grounded outputs contain
box tracks; multiple-choice outputs contain the selected option.

## Development and project records

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for verification and contribution guidance.
Build a distributable package with `uv build`. CI runs the synthetic test suite and
package build without cloud credentials.

Research notes and score tables in [`docs/`](docs/README.md) are dated experiment
records; they do not represent a freshly verified leaderboard. See the
[official challenge](https://eval.ai/web/challenges/challenge-page/2702/overview)
for evaluation details.
