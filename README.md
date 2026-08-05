<p align="center">
  <img src="https://raw.githubusercontent.com/qcri/sayf-eval/main/assets/sayf-eval.png" alt="sayf-eval" width="520">
</p>

<h1 align="center">sayf-eval</h1>

<p align="center">
  <em>A lightweight, model-agnostic framework for evaluating LLMs on cybersecurity benchmarks.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/qcri/sayf-eval/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/sayf-eval/"><img src="https://img.shields.io/pypi/v/sayf-eval.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff">
</p>

---

**sayf** (Arabic: *sword*) **-eval** evaluates any LLM — hosted API or local
checkpoint — through **one common interface**. It rests on two layers kept
separate:

- **Transport — [LiteLLM](https://github.com/BerriAI/litellm):** one `Model`
  adapter for every provider (OpenAI, Anthropic, Azure, …) and any
  OpenAI-compatible local server (vLLM via a `base_url`). No per-provider glue.
- **Structure — [lighteval](https://github.com/huggingface/lighteval)-shaped:**
  a `Task` / `Model` / `Scorer` boundary with a two-level metric split
  (sample-level extract+verdict, corpus-level aggregation).

The **judge is not special** — it is another `Model`, so the model-under-test
and the judge can each be any provider with no code change.

## Why sayf-eval?

General-purpose eval harnesses (lighteval, lm-eval-harness, HELM) are broad by
design and touch cybersecurity only incidentally — usually a single subject such
as MMLU's `computer_security` (which sayf-eval also includes, as `mmlu-cs`).
sayf-eval is instead a **dedicated cybersecurity suite**: 25 sub-tasks across 9
benchmark families spanning cyberthreat intelligence (CWE mapping, CVSS
vulnerability scoring, MITRE ATT&CK technique extraction, threat-actor
attribution), ICS/OT security, security-tooling proficiency, and open-ended CTI
extraction — scored with domain-specific metrics those harnesses don't provide
(CVSS mean-absolute-difference, ATT&CK parent-technique micro-F1, CWE-set
equality, threat-actor alias resolution).

Beyond coverage, it standardizes the *methodology*: fixed, documented decoding
and scoring choices (temperature 0, fixed seed, per-task token budgets,
`<think>` stripping, an explicit denominator policy), and every run writes a
results record that embeds the full pipeline configuration next to the scores —
so entries are comparable by construction, not bare numbers. Scoring is
LLM-as-judge where the judge is just another `Model`, so model-under-test and
judge can each be any provider (hosted API or local vLLM) with no code change.
And as a security benchmark it carries a deliberate dual-use posture: aggregate
scores are safe to publish, while per-sample item text stays private.

## Architecture

```
Task (prompt + params + dataset)
   └─> Model.generate(messages, params)        # LiteLLM under the hood
          └─> Scorer
                judge: Model                    # same type as the model
                extract + verdict (sample level)
                aggregate (corpus level)        # accuracy, VSP MAD, set F1
```

| Module | Role |
|--------|------|
| `sayf_eval/model.py` | `Model` (LiteLLM adapter), `GenParams`, `Response`, concurrent `generate_batch` |
| `sayf_eval/task.py` · `registry.py` | `Sample`, `Task`, the task registry |
| `sayf_eval/judge_prompts.py` | unified judge prompt + per-task format/compare rules |
| `sayf_eval/scorer.py` | judge call, `<think>`-strip, JSON-verdict parsing, skipped handling |
| `sayf_eval/metrics.py` | corpus aggregation (accuracy, ATE micro-F1, VSP MAD) |
| `sayf_eval/datasets.py` · `tasks/` | dataset loaders + task registrations |
| `sayf_eval/pipeline.py` · `cli.py` | end-to-end run loop and CLI |

## Install

```bash
pip install sayf-eval
# or, from source:
pip install -e ".[dev]"
```

## Quick start

```bash
# Credentials: LiteLLM reads provider keys from the environment.
export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...

# End-to-end: inference + judge across a few tasks
sayf-eval run \
  --tasks mcq seceval vsp taa \
  --model openai/gpt-4o \
  --judge anthropic/claude-sonnet-4-20250514 \
  --output-dir outputs/gpt4o \
  --max-samples 5

# Or split the steps
sayf-eval run-inference --tasks mcq --model openai/gpt-4o --output-dir outputs/gpt4o
sayf-eval run-judge     --tasks mcq --judge openai/gpt-4o --output-dir outputs/gpt4o
```

Outputs per task: `<task>_responses.jsonl`, `<task>_detailed.jsonl`, and a
combined `summary.json`.

### Local models (vLLM)

A local model is **just another endpoint**: serve it OpenAI-compatibly and point
sayf-eval at its `base_url`.

```bash
# Serve (tuning flags that used to live in scripts now live at serve time):
vllm serve Qwen/Qwen3-8B --port 8000 --enforce-eager

# Evaluate through the same interface (note the hosted_vllm/ prefix + base-url):
sayf-eval run \
  --tasks mcq vsp \
  --model hosted_vllm/Qwen/Qwen3-8B --base-url http://localhost:8000/v1 --api-key EMPTY \
  --judge anthropic/claude-sonnet-4-20250514 \
  --output-dir outputs/qwen3-8b
```

For reasoning models, pass `--answer-stop` to apply a stop sequence to the answer
*after* the `<think>` block is stripped, and `--max-tokens` to scale the budget.

## Tasks

25 cybersecurity sub-tasks across 9 benchmark families (`sayf-eval run --tasks …`):

- **CTI-Bench:** `mcq`, `rcm`, `vsp`, `ate`, `cti_taa`
- **AthenaBench:** `ckt`, `rms`, `taa`, `athena_ate`, `athena_rcm`, `athena_vsp`
- **SECURE:** `secure_maet`, `secure_cwet`, `secure_kcv`
- **RedSage:** `redsage_frameworks`, `redsage_generals`, `redsage_skills`, `redsage_cli`, `redsage_kali`
- **Other MCQ:** `seceval`, `cybermetric`, `secbench`, `mmlu-cs`, `cissp`
- **SEvenLLM:** `sevenllm` (open-ended structured CTI extraction / analysis, judged semantically)

`cissp` needs a dataset path via `SAYF_EVAL_CISSP_PATH` (not a public dataset);
all others load from HuggingFace / GitHub on first run.

## Standardized pipeline choices

sayf-eval applies fixed, documented choices that remove measurement artifacts
without changing task semantics: temperature 0 / top_p 1 / fixed seed; per-task
token budgets; `<think>` stripped before judging with the stop sequence applied
to the answer only; and **denominator = all attempted items** (unparseable/empty
answers are incorrect; only judge-API failures are excluded — from both
numerator and denominator).

## Results & leaderboard

Every `sayf-eval run` writes a canonical **results record** to
`<output-dir>/results/<model>/results_<ts>.json`. Because scores are
pipeline-dependent, the record embeds the **full pipeline configuration**
(decoding params, token-budget policy, `<think>` handling, denominator policy,
judge model) next to the per-task metrics — so entries are comparable by
construction, not bare numbers.

### The results record

`results_<ts>.json` is a single JSON object (record schema `1.0`):

```json
{
  "model": { "name": "openai/gpt-4o", "provider": "openai", "base_url": null },
  "judge": { "name": "anthropic/claude-sonnet-4-20250514", "provider": "anthropic", "base_url": null },
  "pipeline": {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "max_tokens": "per-task-calibrated",
    "max_tokens_override": null,
    "answer_stop": null,
    "think_handling": "strip <think>...</think> before judging, then apply stop sequence to the answer only",
    "scoring": "llm-as-judge: single call performs extraction + CORRECT/INCORRECT verdict",
    "denominator_policy": "accuracy = correct / total over all attempted items; unparseable/empty answers count as incorrect; only judge-API failures are excluded (skipped) from both numerator and denominator"
  },
  "results": {
    "mcq": { "accuracy": 0.667, "correct": 2, "total": 3, "skipped": 0 },
    "vsp": { "accuracy": 0.5, "correct": 1, "total": 2, "skipped": 0, "mad": 1.3 }
  },
  "tasks": ["mcq", "vsp"],
  "task_sources": {
    "mcq": { "type": "hf_dataset", "dataset_name": "CTI-Bench MCQ", "hf_repo": "RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "subset": "cti-mcq", "split": "test" }
  },
  "sayf_eval_version": "0.1.2",
  "schema_version": "1.0",
  "created_at": "2026-01-01T00:00:00+00:00"
}
```

| Field | Meaning |
|-------|---------|
| `model` / `judge` | The model-under-test and the judge, each a `{name, provider, base_url}` triple. `name` is the LiteLLM model string; `provider` is its prefix; `base_url` is set for local / self-hosted endpoints. |
| `pipeline` | The exact decoding + scoring configuration that produced the scores: `temperature` / `top_p` / `seed`; the `max_tokens` budget policy (`"per-task-calibrated"` or `"override"`, with the value in `max_tokens_override`); `answer_stop`; `think_handling`; `scoring`; and the `denominator_policy`. This is what makes records comparable by construction. |
| `results` | Per-task metrics. Every task carries `accuracy` / `correct` / `total` / `skipped`; VSP tasks add `mad` (CVSS mean-absolute-difference), ATE tasks add `precision` / `recall` / `f1` (parent-technique micro-average). |
| `tasks` | Sorted list of the task names present in `results`. |
| `task_sources` | Per-task dataset provenance — `hf_dataset` (`hf_repo` + `subset` + `split`), `url`, or `other` — so the record is self-describing. |
| `sayf_eval_version` | The sayf-eval version that produced the record. |
| `schema_version` | Version of the record schema itself (currently `1.0`). |
| `created_at` | UTC timestamp (ISO-8601) of when the record was written. |

The record carries metrics and configuration only — never prompt, gold-answer,
or model-response text — so the scores artifact is safe to publish while
per-sample details stay private.

Optionally push to a HuggingFace dataset (`pip install 'sayf-eval[hub]'`):

```bash
sayf-eval run --tasks mcq vsp --model openai/gpt-4o --judge openai/gpt-4o \
  --output-dir outputs/gpt4o \
  --results-org my-org --push-scores      # private dataset by default
```

**Security posture (this is a cybersecurity benchmark):**

| What | Flag | Visibility |
|------|------|-----------|
| Scores record (metrics + pipeline config, **no item text**) | `--push-scores` | private; `--public` to publish |
| Per-sample details (prompt / gold / response) | `--push-details` | **always private** (benchmark-leakage / dual-use) |

Nothing is pushed without an explicit flag. The scores artifact never contains
prompt or answer text, so it is safe to make public; details stay private
regardless of `--public`.

### Community leaderboard (HF Community-Evals)

Level 2 emits the two artifacts HuggingFace aggregates into a rendered
leaderboard ([docs](https://huggingface.co/docs/hub/eval-results)) — opt-in, and
a deliberate disclosure decision for security tasks.

```bash
# 1. Benchmark spec: register sayf-eval as a HF benchmark dataset (one
#    sub-leaderboard per task). Writes eval.yaml; --push-to creates the dataset.
sayf-eval benchmark-spec --out eval.yaml            # all 25 tasks
sayf-eval benchmark-spec --push-to qcri/sayf-eval --public   # private without --public

# 2. Per-model results: turn a results record into .eval_results/*.yaml and
#    (optionally) open a community PR to the model repo so scores show on its card.
sayf-eval eval-results --results outputs/gpt4o/results/openai__gpt-4o/results_*.json \
  --benchmark-id qcri/sayf-eval --out .eval_results/sayf-eval.yaml \
  --submit-pr openai/gpt-4o
```

Each `.eval_results` entry carries the pipeline config in its `notes`, so the
public leaderboard never shows a bare number. Two one-time HF steps are required
to go live (both noted by the CLI): the `sayf-eval` `evaluation_framework` must be
added to HF's enum, and the benchmark dataset allow-listed (registration is beta).

## Development

```bash
make install     # pip install -e ".[dev]"
make style       # ruff format + ruff check --fix
make quality     # ruff format --check + ruff check  (CI gate)
make test        # pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

## License

[MIT](LICENSE).
