# seceval-harness — project context

A lightweight, model-agnostic framework for cybersecurity LLM benchmark
evaluation. It re-shapes our existing script-based harness into a small reusable
framework: **LiteLLM** as the universal transport (talk to any provider the same
way), and a **lighteval-shaped** `Task` / `Model` / `Scorer` structure on top.

The full design intent is frozen in **[PROPOSAL.md](PROPOSAL.md)** — read it
first. The actionable, living breakdown is in **[PLAN.md](PLAN.md)**.

## Working agreement (status tracking)

- **PLAN.md is the single source of truth for status.** It is a checkbox list.
- After completing any unit of work, **update the corresponding `- [ ]` to
  `- [x]` in PLAN.md** in the same change. Keep checkbox text stable; only flip
  the box. Add new `- [ ]` items when scope is discovered rather than silently
  doing untracked work.
- **PROPOSAL.md is frozen design intent** — do not edit it to reflect progress.
  If the design itself changes, note the deviation in PLAN.md and call it out;
  only amend PROPOSAL.md on an explicit decision to revise the design.
- CLAUDE.md (this file) holds durable context only — update it when locations,
  references, or architectural invariants change, not for task progress.

## Architecture (two layers, kept separate)

1. **Transport — LiteLLM.** One model adapter, not a provider matrix. Unified
   inference params become a single config object passed to LiteLLM. A local
   model is just another endpoint (vLLM via `base_url`). The judge is not
   special — it is another `Model` call with its own prompt + params.
2. **Structure — lighteval-shaped (mirror, don't depend, for now).** Backend
   abstraction for the `Model` boundary; two-level metrics: **sample-level**
   (judge/extract) + **corpus-level** (aggregate score).

Model-under-test and judge are the **same `Model` type**, swappable to any
provider via LiteLLM with no code change.

## Key locations

- **This repo:** `/export/home/aberriche/BenchBench/seceval-harness/` (fresh, `main`).
- **Original harness (logic to port, leave untouched):**
  `/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/` — esp.
  `unified-benchmark-pipeline/{run_inference_benchmarks.py, run_evaluate_llm_judge.py, evaluate.py}`.
- **References (sibling, NOT committed here):** `/export/home/aberriche/BenchBench/_refs/`
  - `lighteval/` — pinned **v0.10.0** (`4d47029`). Structural reference only.
    Key files: `src/lighteval/models/abstract_model.py`,
    `src/lighteval/models/endpoints/litellm_model.py`,
    `src/lighteval/metrics/metrics_sample.py`, `metrics_corpus.py`,
    `src/lighteval/tasks/lighteval_task.py`.
  - `litellm/` — shallow clone (`aaf1e24`). Reference for `completion()`
    signatures and provider param maps.

## Invariants to preserve when porting (paper's failure-mode fixes)

These are load-bearing for the original audit's claims — the refactor must not
silently drop them:

- **Chat template** applied via the model's native template (fixes template
  incompatibility). LiteLLM does server-side templating; the local/vLLM path
  must still handle this explicitly.
- **`<think>` / reasoning handling:** strip the thinking block, then apply stop
  sequences to the answer portion only (the RedSage 86-point fix).
- **Denominator policy:** all attempted items count; unparseable/empty responses
  are incorrect (no valid-only denominator inflation). Judge-API-failure samples
  are excluded from *both* numerator and denominator.
- **Per-task scoring rules** (sample-level): MCQ letter, multi-answer set match,
  CWE-ID sets, CVSS vector + MAD, MITRE T-ID/M-ID sets, alias-aware threat-actor
  matching. Currently encoded as per-task `format_hint` + `compare_rule` in the
  judge prompt.

## What LiteLLM does NOT subsume (still ours to build)

- In-process / offline vLLM batch generation for local-model throughput.
- The chat-template + `<think>`-strip + post-gen stop-sequence logic above.
- All cybersecurity-specific prompt, judge, and scoring content.

## Scope

- **In:** generative + LLM-judge scoring across our benchmark families; possibly
  logprob/MCQ scoring (open question — it is the recommended mitigation for the
  logprob-vs-generative failure mode).
- **Out (for now):** agentic/tool-use/CTF + sandboxing (future path: Inspect AI),
  multi-turn, leaderboard hosting.
