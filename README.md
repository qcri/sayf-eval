<p align="center">
  <img src="assets/sayf-eval.png" alt="sayf-eval" width="520">
</p>

<h1 align="center">sayf-eval</h1>

<p align="center">
  <em>A lightweight, model-agnostic framework for evaluating LLMs on cybersecurity knowledge benchmarks.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/qcri/sayf-eval/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
 <a href="https://pypi.org/project/sayf-eval/"><img src="https://img.shields.io/pypi/v/sayf-eval.svg" alt="PyPI"></a>
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="https://img.shields.io/badge/code%20style-ruff-e6e6e6.svg">
  <source
    media="(prefers-color-scheme: light)"
    srcset="https://img.shields.io/badge/code%20style-ruff-000000.svg">
  <img
    src="https://img.shields.io/badge/code%20style-ruff-000000.svg"
    alt="Ruff">
</picture>
</p>

<p align="center">
  <a href="https://github.com/qcri/sayf-eval/actions/workflows/release.yaml"><img src="https://github.com/qcri/sayf-eval/actions/workflows/release.yaml/badge.svg" alt="Release"></a>
  <a href="https://github.com/qcri/sayf-eval/actions/workflows/quality.yaml"><img src="https://github.com/qcri/sayf-eval/actions/workflows/quality.yaml/badge.svg" alt="Quality"></a>
  <a href="https://github.com/qcri/sayf-eval/actions/workflows/tests.yaml"><img src="https://github.com/qcri/sayf-eval/actions/workflows/tests.yaml/badge.svg" alt="Tests"></a>
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

sayf-eval is not a general harness and does not try to be one. Use it when you are evaluating models on cybersecurity and need the score to mean something. sayf-eval exists for a critical problem:

> In cybersecurity, the same model on the same dataset can score wildly
> differently depending on **how** the evaluation is run.

We ran 8 cybersecurity benchmarks on 10 models. Some of what we found:

- A benchmark's stop sequence fired **inside the model's own reasoning**, so it
  returned empty answers. Fixing it: **+86 points**.
- A benchmark capped output at 5 tokens — below the API's 16-token minimum — so
  every request failed silently and looked like a bad model. Fixing it:
  **+81 points**.
- Dropping unparseable answers instead of marking them wrong turned **0.2%
  into 100%** on one task.
- Two benchmarks score the same CVSS task in **opposite directions** (lower is
  better vs higher is better), so they disagree about which model is best.
- Scoring by log-probability instead of the generated answer moved one model
  from **45.7% to 86.6%**.

None of this measures security knowledge — it measures the harness. In total,
**9 of 10 models moved at least 3 ranks _on at least one benchmark_** once
these were fixed.

A general harness will not fix this for you, because the broken choices live
inside each benchmark's own scripts and are mostly undocumented. sayf-eval makes
**one set of choices**, applies it to **every task**, and writes it into
**every results file**.

### How it compares

| | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | [lighteval](https://github.com/huggingface/lighteval) | **sayf-eval** |
|---|---|---|---|
| **Scope** | general (60+ benchmarks) | general (1000+ tasks) | cybersecurity only (23 tasks, 8 families) |
| **Coverage of the domain** | 1 MCQ subject, 100 questions | 1 MCQ subject, 100 questions | knowledge **and** analytical tasks: CVSS scoring, ATT&CK extraction, root-cause mapping, attacker attribution, open-ended CTI analysis |
| **Cyber benchmarks built in** | `mmlu_computer_security` | `mmlu:computer_security` | that plus CTI-Bench, AthenaBench, SECURE, RedSage, SecEval, CyberMetric, SecBench |
| **Cyber metrics** (CVSS error, ATT&CK set-F1, attacker aliases) | write your own | write your own | built in |
| **Open-ended answers** | exact / log-prob match | exact match, or a custom metric you wire up | LLM judge is a first-class `Model` — change provider, not code |
| **Reasoning models (`<think>`)** | you handle it | you handle it | stripped before judging; stop sequence applied to the answer only |
| **Unparseable / empty answers** | up to each task | up to each task | one fixed policy: counted wrong, rate reported |
| **Token budgets** | you set them per task | you set them per task | pinned per task, so nothing truncates silently |
| **Local + hosted models** | many backends, different paths | many backends | one LiteLLM path; vLLM is just a `base_url` |
| **What the results file records** | metrics + run config | metrics + run config | metrics + **full pipeline config** (decoding, budgets, denominator, judge) |
| **Dual-use handling** | not a concern | not a concern | scores publishable; per-item prompts and answers stay private |
| **Defaults come from** | community task configs | community task configs | an audit of 8 benchmarks × 10 models, 15 documented failure modes |

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

24 cybersecurity sub-tasks across 9 benchmark families (`sayf-eval run --tasks …`):

- **CTI-Bench:** `mcq`, `rcm`, `vsp`, `ate`, `cti_taa`
- **AthenaBench:** `ckt`, `rms`, `taa`, `athena_ate`, `athena_rcm`, `athena_vsp`
- **SECURE:** `secure_maet`, `secure_cwet`, `secure_kcv`
- **RedSage:** `redsage_frameworks`, `redsage_generals`, `redsage_skills`, `redsage_cli`, `redsage_kali`
- **Other MCQ:** `seceval`, `cybermetric`, `secbench`, `mmlu-cs`
- **SEvenLLM:** `sevenllm` (open-ended structured CTI extraction / analysis, judged semantically)

All tasks load from HuggingFace / GitHub on first run.

## Standardized pipeline choices

sayf-eval applies fixed, documented choices that remove measurement artifacts
without changing task semantics: temperature 0 / top_p 1 / fixed seed; per-task
token budgets; `<think>` stripped before judging with the stop sequence applied
to the answer only; and **denominator = all attempted items** (unparseable/empty
answers are incorrect; only judge-API failures are excluded — from both
numerator and denominator).

## Leaderboard

Ten cybersecurity LLMs across **24 sub-tasks** (9 benchmark families), each cell
scored by a single `gpt-5.4` judge under the unified extract-and-verdict prompt
(temperature 0, top_p 1, seed 42). Ranked by mean strict-verdict accuracy:

| Rank | Model | Avg accuracy (%) |
|---:|---|---:|
| 1 | `claude-sonnet-4-6` | 76.0 |
| 2 | `gpt-5.4` | 73.4 |
| 3 | `gemma-4-31B-it` | 69.5 |
| 4 | `Qwen/Qwen3.6-35B-A3B` | 65.0 |
| 5 | `Llama-Primus-Nemotron-70B-Instruct` | 64.7 |
| 6 | `RISys-Lab/RedSage-Qwen3-8B-DPO` | 64.1 |
| 7 | `Llama-3.3-70B-Instruct` | 62.8 |
| 8 | `openai/gpt-oss-20b` | 61.9 |
| 9 | `fdtn-ai/Foundation-Sec-8B-Instruct` | 57.4 |
| 10 | `trendmicro-ailab/Llama-Primus-Merged` | 54.7 |

**→ Full per-task table + provenance: [`leaderboard/`](leaderboard/README.md).**
Each model is one standard results record (schema 1.1, aggregate-only) under
[`leaderboard/results/`](leaderboard/results/); regenerate the table with
`python leaderboard/render_table.py leaderboard`.

The records follow the standard sayf-eval schema, which maps onto Every Eval
Ever's (EEE) `EvaluationLog` schema (one log per task) for HF Community-Evals; a
sayf-eval converter for EEE lives in the
[every_eval_ever](https://github.com/evaleval/every_eval_ever) project.

## Results records

Every `sayf-eval run` writes a canonical **results record** to
`<output-dir>/results/<model>/results_<ts>.json`. Because scores are
pipeline-dependent, the record embeds the **full pipeline configuration**
(decoding params, token-budget policy, `<think>` handling, denominator policy,
judge model) next to the per-task metrics — so entries are comparable by
construction, not bare numbers.

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
