# Plan: seceval-harness — lightweight cybersecurity eval framework

## Context

We are turning the existing script-based cybersecurity benchmark harness
(`BenchmarkingSecBenchmarks/unified-benchmark-pipeline/`) into a small, reusable
framework with **one model-agnostic interface for all LLMs**. The design (frozen
in `PROPOSAL.md`) is two layers kept separate:

- **Transport — LiteLLM**: one adapter, every provider via the OpenAI-style call
  format; local models are just a `base_url`. Judge is not special — it's
  another `Model` call.
- **Structure — lighteval-shaped (mirror, don't depend)**: a `Model` backend
  boundary and a two-level metric split (sample-level extract/verdict +
  corpus-level aggregation).

Today the harness hand-rolls a provider matrix (`chat_completion_api`,
`anthropic_messages_api`, `api_style` dispatch) and bundles inference, judging,
and scoring across three large scripts. We replace the transport with LiteLLM
and re-shape the cyber-specific logic behind thin `Model`/`Task`/`Scorer`
interfaces, **preserving the paper's failure-mode fixes**.

This file is the living status tracker. Flip `- [ ]` → `- [x]` as work completes
(per `CLAUDE.md`).

## Locked decisions

- **Local serving:** vLLM **server** behind LiteLLM. Run `vllm serve` (OpenAI-
  compatible) and point LiteLLM at its `base_url`. **One unified `Model` type**
  for API + local — no torch/vLLM import in the framework core. vLLM launch
  flags (`min_tokens`, `num_gpu_blocks_override`, `enforce_eager`, max-len) move
  to serve time and are documented, not coded into the framework.
- **Scoring:** generative + LLM-judge only (extract + verdict in one judge
  call). No logprob/MCQ path for now (boundary left open via the metric `kind`).
- **MVP scope:** representative subset first — `mcq`, `seceval` (set match),
  `vsp` (CVSS MAD), `taa` (alias-aware) — then fan out to all ~24 tasks.
- **lighteval:** mirror its shape with our own thin classes; do **not** add it as
  a dependency. Keep the boundary compatible for later adoption.
- **Per-benchmark variation:** lives in a **task registry** (data, not subclasses).

## Reference shapes being mirrored (read-only refs in `../_refs/`)

- `lighteval/.../models/endpoints/litellm_model.py` — `LiteLLMClient`: builds a
  `kwargs` dict for `litellm.completion(**kwargs)`, `ThreadPoolExecutor`
  concurrency (`__call_api_parallel`), exponential-backoff retries, content-
  filter → empty response, reasoning-model token bump (`_prepare_max_new_tokens`).
- `lighteval/.../metrics/utils/metric_utils.py` — `Metric(sample_level_fn,
  corpus_level_fn)`; `compute_sample()` returns `{name: score}`,
  `get_corpus_aggregations()` returns the corpus fn. We mirror this two-level split.
- `litellm` `completion()` — `model` (provider prefix routing: `anthropic/…`,
  `openai/…`, `azure/…`, `hosted_vllm/…` or `openai/…`+`base_url` for local),
  `messages`, `temperature`, `max_tokens`/`max_completion_tokens`, `top_p`,
  `stop`, `seed`, `response_format` (JSON/schema for the judge), `timeout`,
  `num_retries`. Response: `resp.choices[0].message.content`, `resp.usage`.

## Module layout (`seceval-harness/`)

```
seceval-harness/
├── pyproject.toml                # deps: litellm, datasets, cvss, pydantic
├── PROPOSAL.md  CLAUDE.md  PLAN.md
├── seceval/
│   ├── model.py        # Model (LiteLLM), GenParams, Response, parallel generate
│   ├── task.py         # Task dataclass + Sample type
│   ├── registry.py     # TASKS registry: per-benchmark config (prompt, loader, scorer kind, budget)
│   ├── scorer.py       # JudgeScorer: build prompt -> judge Model -> parse -> sample verdict
│   ├── judge_prompts.py# per-task format_hint / compare_rule (ported verbatim)
│   ├── metrics.py      # sample_metric + corpus_metric (accuracy, VSP MAD, set P/R/F1)
│   ├── datasets.py     # tsv/jsonl/HF loaders + collect_* per benchmark
│   ├── pipeline.py     # run loop: Task -> Model.generate -> Scorer -> aggregate -> JSONL
│   └── cli.py          # run-inference / run-judge / run (end-to-end) entrypoints
├── configs/            # example model + judge configs (.env.example, gen params)
└── tests/              # unit tests per layer + a tiny smoke eval
```

## Behavior to port (source → destination), with invariants preserved

| Source (unified-benchmark-pipeline) | Destination | Notes / invariant |
|---|---|---|
| `chat_completion_api`/`anthropic_messages_api`/`generate_response` dispatch (`run_inference_benchmarks.py:207–348`, `evaluate.py:174–237`) | `model.py` (single LiteLLM adapter) | **Delete the provider matrix.** Retry/backoff + content-filter→empty mirrors `litellm_model.py` `__call_api`. |
| `generate_responses_vllm` min_tokens=50, `initialize_vllm` flags (`run_inference_benchmarks.py:351–443`) | **serve-time docs** (not code) | min_tokens / gpu_blocks_override / enforce_eager become `vllm serve` flags; documented in README. |
| `apply_chat_template` (`run_inference_benchmarks.py:312,326`) | not in core | vLLM server applies the model's chat template; framework sends `messages`. |
| `<think>` strip + post-gen stop on answer only (`run_evaluate_llm_judge.py:686`) | `scorer.py` post-processing | **Backend-agnostic**: strip `</think>` then apply stop to answer portion before judging. Preserves the RedSage 86-pt fix. |
| `get_task_type` map (`run_inference_benchmarks.py:109–149`) | `registry.py` `task_type` | one entry per task. |
| dataset loaders `load_tsv/jsonl`, `collect_*` (`run_inference_benchmarks.py:78–106,505–652,655–1335`) | `datasets.py` | port loaders for subset tasks first; sample fields normalized into `Sample`. |
| system prompts / templates (CTI, CyberMetric, SecEval 1-shot, MMLU 5-shot) (`run_inference_benchmarks.py:52–63,563–565`; `evaluate.py:833–840`) | `registry.py` per task | content unchanged. |
| per-task token budgets / `--max_tokens_config` | `registry.py` `max_tokens` + optional override file | calibrated budgets. |
| `create_judge_prompt` format_hint/compare_rule (`run_evaluate_llm_judge.py:86–254`) | `judge_prompts.py` | ported verbatim per task_type. |
| `parse_judge_response` JSON+regex fallback, skipped handling (`run_evaluate_llm_judge.py:257–345`) | `scorer.py` | **Invariant:** judge-API-failure → `skipped` (out of numerator AND denominator). |
| `calculate_vsp_mad`, `compute_ate_metrics`, `compute_vsp_metrics`, `parse_ids_from_text`, `_parent_only` (`run_evaluate_llm_judge.py:348–394,871–931`; `evaluate.py:346–372`) | `metrics.py` | CVSS via `cvss.CVSS3`; ATE micro-F1 parent-only; set P/R/F1. |
| corpus accuracy = correct/total, all attempted counted (`run_evaluate_llm_judge.py:624–634`) | `metrics.py` `score_corpus` | **Invariant:** unparseable/empty = incorrect; no valid-only denominator. |

## Execution checklist

### Phase 0 — Scaffolding
- [x] Copy plan to `PLAN.md` (this file, checkbox tracker)
- [x] `pyproject.toml` (litellm, datasets, cvss, pydantic), `.gitignore`, package skeleton `seceval/`
- [ ] Commit context docs + scaffold (PROPOSAL.md, CLAUDE.md, PLAN.md)

### Phase 1 — Framework core
- [x] `model.py`: `GenParams`, `Response`, `Model` over `litellm.completion` (provider-prefix routing, `base_url` for vLLM)
- [x] `model.py`: `generate_batch` with `ThreadPoolExecutor` + retries/backoff + content-filter→`ok=False`
- [x] `task.py` + `registry.py`: `Sample`, `Task`, empty `TASKS` registry
- [x] `judge_prompts.py`: unified judge prompt + format_hint/compare_rule (all task_types ported)
- [x] `scorer.py`: `JudgeScorer.score_sample` (build prompt → judge Model → `parse_judge_response` port; `<think>` strip + skipped handling)
- [x] `metrics.py`: accuracy (correct/total), set match, `calculate_vsp_mad`, alias note for TAA
- [x] `pipeline.py`: end-to-end loop → responses JSONL + `*_detailed.jsonl` + `summary.json`
- [x] `cli.py`: `run-inference`, `run-judge`, `run` subcommands
- [x] Unit tests for model (mock litellm), scorer parsing (incl. skipped), metrics (VSP MAD, set F1) — 23/23 pass via `tests/run_no_pytest.py` (pytest install blocked on login node)

### Phase 2 — MVP tasks wired to registry
- [x] Port loaders + registry entries: `mcq`, `seceval`, `vsp`, `taa`
- [x] Smoke eval: offline end-to-end (mock model + judge) verified, **and live** — SLURM job 311002 (`cpu-all`) ran `pytest` (23/23) + an end-to-end Azure gpt-5.4 round-trip (model + judge) on real `mcq`/`seceval` data: correct denominator, sane accuracy, `RESULT: PASS`. Harness: `tests/live_smoke_azure.py`, `slurm/smoke_azure.sh`.
- [x] Live local-vLLM path (`slurm/smoke_local_vllm.sh`): **PASS** — job 311007 (`gpu-all`, A16) served Qwen2.5-0.5B via vLLM and ran the full pipeline with `hosted_vllm/…` + `base_url`, model **and** judge = the local server; mcq 3/3, seceval 2/3, correct denominators, detailed JSONL emitted. (Earlier 310999/311004 failed only because `gpu-short` assigns Tesla P100 sm_60, unsupported by the env's torch/vLLM — exclude P100 nodes: `--exclude=crimv3mgpu003,crimv3mgpu021,crimv3mgpu022,crimv3mgpu023,crimv3srv040,crimv3srv041,crimv3srv042,crimv3srv047`.)
- [x] README quick-start (incl. `vllm serve` example + base_url)

### Phase 3 — Fan-out to remaining tasks
- [x] Port remaining MCQ-family tasks (`cybermetric`, `cissp`, `mmlu-cs`, `secbench`, `secure_*`, `redsage_*`, `ckt`) — via `make_hf_loader` / dedicated loaders
- [x] Port remaining structured tasks (`rcm`/`athena_rcm`, `ate`/`athena_ate`, `rms`, `athena_vsp`, `cti_taa`) — via `make_hf_loader` / `make_athena_loader`
- [x] 24 tasks registered; all 9 task_types resolve a judge prompt; 29/29 unit tests pass; package compiles
- [ ] Parity check vs original harness on a fixed sample set (accuracy within tolerance) — run on a compute node (needs deps + dataset network)

## Verification

- **Unit:** `pytest tests/` — model adapter (mocked `litellm.completion`), judge
  parsing incl. malformed JSON + `ERROR:`/empty → `skipped`, VSP MAD on known
  vectors, set-match P/R/F1.
- **Smoke (end-to-end):** `seceval run --tasks mcq seceval vsp taa
  --model openai/gpt-... --judge ... --max-samples 5` against an API judge;
  confirm JSONL + `summary.json` emitted, denominator = total attempted.
- **Local path:** start `vllm serve <model>`, run `seceval run --model
  openai/<served-name> --base-url http://localhost:8000 ...`; confirm a local
  model routes through the same `Model` with no code change.
- **Parity:** run the 4 MVP tasks on a fixed sample subset through both the old
  harness and the new framework with the same judge; per-task accuracy should
  match within a small tolerance (justify any gap by the documented
  `<think>`/denominator handling).
