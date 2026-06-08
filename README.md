# seceval-harness

A lightweight, **model-agnostic** framework for cybersecurity LLM benchmark
evaluation. One interface for every LLM: **LiteLLM** as the universal transport,
a **lighteval-shaped** `Task` / `Model` / `Scorer` structure on top. The judge is
not special — it is another `Model`, so the model-under-test and the judge can be
any provider (hosted API or local vLLM) with no code change.

See [PROPOSAL.md](PROPOSAL.md) for the design rationale, [PLAN.md](PLAN.md) for
the build status (checkbox tracker), and [CLAUDE.md](CLAUDE.md) for context.

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
| `seceval/model.py` | `Model` (LiteLLM adapter), `GenParams`, `Response`, concurrent `generate_batch` |
| `seceval/task.py` · `registry.py` | `Sample`, `Task`, the task registry |
| `seceval/judge_prompts.py` | unified judge prompt + per-task format/compare rules |
| `seceval/scorer.py` | judge call, `<think>`-strip, JSON-verdict parsing, skipped handling |
| `seceval/metrics.py` | corpus aggregation (accuracy, ATE micro-F1, VSP MAD) |
| `seceval/datasets.py` · `tasks/` | dataset loaders + task registrations |
| `seceval/pipeline.py` · `cli.py` | end-to-end run loop and CLI |

## Install

```bash
pip install -e .          # litellm, datasets, cvss, pydantic
```

## Quick start

```bash
# 1. Credentials (LiteLLM reads provider keys from the environment)
cp configs/.env.example .env && set -a && . ./.env && set +a

# 2. End-to-end: inference + judge across the MVP tasks
seceval run \
  --tasks mcq seceval vsp taa \
  --model openai/gpt-4o \
  --judge anthropic/claude-sonnet-4-20250514 \
  --output-dir outputs/gpt4o \
  --max-samples 5

# Or split the steps
seceval run-inference --tasks mcq --model openai/gpt-4o --output-dir outputs/gpt4o
seceval run-judge     --tasks mcq --judge openai/gpt-4o --output-dir outputs/gpt4o
```

Outputs per task: `<task>_responses.jsonl`, `<task>_detailed.jsonl`, and a
combined `summary.json`.

### Local models (vLLM)

Local serving is **vLLM behind LiteLLM**: serve the model OpenAI-compatibly and
point the framework at its `base_url` — it is just another `Model`.

```bash
# Serve (tuning flags that used to live in the harness now live here):
vllm serve Qwen/Qwen3-8B \
  --port 8000 \
  --enforce-eager            # + --num-gpu-blocks-override / max-len as needed

# Evaluate through the same interface (note openai/ prefix + base-url):
seceval run \
  --tasks mcq vsp \
  --model openai/Qwen/Qwen3-8B --base-url http://localhost:8000/v1 --api-key EMPTY \
  --judge anthropic/claude-sonnet-4-20250514 \
  --output-dir outputs/qwen3-8b
```

For reasoning models, pass `--answer-stop` to apply a stop sequence to the
answer *after* the `<think>` block is stripped (the RedSage fix).

## Tasks

MVP (current): `mcq` (CTI-Bench), `seceval`, `vsp` (CVSS MAD), `taa`
(alias-aware). The remaining ~20 CTI-Bench / AthenaBench / SECURE / RedSage /
SecBench tasks are ported in Phase 3 (see PLAN.md).

## Standardized pipeline choices

Carried over from the original audit, these remove measurement artifacts without
changing task semantics: temperature 0 / top_p 1 / fixed seed; per-task
calibrated token budgets; `<think>` stripped before judging with the stop
sequence applied to the answer only; **denominator = all attempted items**
(unparseable/empty answers are incorrect; only judge-API failures are excluded,
from both numerator and denominator).

## Testing

```bash
pytest tests/
```

> **Cluster note:** the GPU login node blocks `pip install` and `pytest` install
> (the same resource guard that gates heavy I/O — submit installs via SLURM, or
> use a prepared env). To validate the core without pytest:
> ```bash
> PYTHONPATH=.:.deps python tests/run_no_pytest.py
> ```
> (`.deps` is an optional local dir for a directly-extracted `cvss` wheel when
> pip is unavailable.)
