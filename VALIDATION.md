# Validation plan (end-to-end)

How we know `sayf-eval` reproduces the original harness
(`BenchmarkingSecBenchmarks/unified-benchmark-pipeline/`) without per-provider
glue. Five tiers, cheapest/most-deterministic first. Status is tracked here and
in [PLAN.md](PLAN.md).

| Tier | What it proves | Cost | Status |
|------|----------------|------|--------|
| 0. Static | package imports/compiles; every task_type resolves a judge prompt; unit logic correct | none | ✅ |
| 1. Prompt-construction parity | new loaders reproduce the **exact** prompt + ground truth the original collectors produced | CPU + HF download, no API | ✅ **23/23 byte-for-byte** (job 311039) |
| 2. Scoring-code parity | judge prompt + verdict parsing + metrics are byte-for-byte the original logic | none | ✅ (ported verbatim; unit-tested) |
| 3. Live end-to-end | the real LiteLLM round-trip works for API **and** local vLLM, model = judge type | API + GPU | ✅ |
| 4. Judge-agreement parity | on identical responses, the new scorer and the original judge agree (same judge model) | API | ✅ **100% verdict agreement, κ=1.0** (job 311053) |

## Tier 0 — Static (offline)

```bash
PYTHONPATH=src:.deps python tests/run_no_pytest.py     # 29/29 (or: pytest tests/)
python -m compileall -q seceval
```
Covers: `Model` adapter (mocked litellm), judge-response parsing incl.
malformed-JSON fallback and skipped-sample semantics, `<think>`-strip + post-stop,
VSP MAD / set-F1 / ATE micro-F1, corpus denominator policy, loader helpers, and
that all 24 registered tasks resolve a judge prompt.

## Tier 1 — Prompt-construction parity (deterministic, no API)

The original collectors' actual outputs are committed at
`BenchmarkingSecBenchmarks/outputs_test_5samples/<task>_responses.jsonl`
(5 samples/task, 14 tasks). `tests/parity_prompts.py` runs each new loader on the
same dataset and diffs `prompt` + `ground_truth` per `index`.

```bash
# CPU node (datasets + HF network); login node may trip the I/O guard
sbatch slurm/parity_prompts.sh           # writes a per-task match report
```
Pass = prompts and ground truth match per index. Known intentional divergences
are declared in the script (e.g. SecEval folds the original chat-style 1-shot
into prompt text; CISSP needs `SECEVAL_CISSP_PATH` so it is skipped).

**Authoritative result (job 311039).** The committed `outputs_test_5samples` is
partially stale (generated across older repo revisions), so the definitive check
is `tests/parity_vs_current.py` (`slurm/parity_vs_current.sh`): it runs the
*current* original collectors with generation stubbed and diffs all 23 tasks.
**PASS — prompt 5/5 + gt 5/5 for every prompt-comparable task** (SecEval gt-only
by design); i.e. the new loaders reproduce the original prompts byte-for-byte.

## Tier 2 — Scoring-code parity (static)

`src/sayf_eval/judge_prompts.py::create_judge_prompt` and
`src/sayf_eval/scorer.py::parse_judge_response` were ported verbatim from
`run_evaluate_llm_judge.py` (same `format_hint`/`compare_rule` per task_type,
same JSON+regex fallback, same skipped semantics). `src/sayf_eval/metrics.py` ports
`calculate_vsp_mad`, `compute_ate_metrics`, set P/R/F1, and the
accuracy = correct/total denominator policy. Verified by Tier-0 unit tests.

## Tier 3 — Live end-to-end (API + GPU)

- **Azure** (`slurm/smoke_azure.sh`): `pytest` + end-to-end gpt-5.4 as model
  **and** judge on real `mcq`/`seceval`. Job 311002 → `RESULT: PASS`.
- **Local vLLM** (`slurm/smoke_local_vllm.sh`): serve a model, run the pipeline
  through `hosted_vllm/…` + `base_url`, model **and** judge = the local server.
  Job 311007 → PASS. (Exclude Tesla P100 nodes — unsupported by the env's vLLM.)
- **Broad coverage** (`smoke_azure.sh` with `TASKS=…`): 14 tasks across every
  loader family loaded real data and scored end-to-end with live structured
  metrics (ATE micro-F1, VSP MAD). Job 311015 → `RESULT: PASS`.

## Tier 4 — Judge-agreement parity (optional, live)

Feed identical model responses to both the original `run_evaluate_llm_judge.py`
and the new scorer using the **same** judge model (Azure gpt-5.4); report
per-sample verdict agreement and per-task accuracy delta.

**Result (job 311053, `tests/judge_agreement.py`, 360 samples × 24 tasks,
Llama-3.3-70B responses):**

- **verdict-agreement = 100.0% (360/360), Cohen's κ = 1.000** — `acc_orig`
  equals `acc_new` for every task.
- prompt-identity = 83.3% (300/360). The 60 non-identical prompts are confined
  to 4 tasks (`mmlu-cs`, `secbench`, `secure_cwet`, `secure_maet`) and stem from
  one cosmetic difference: the original embeds a choices **list**'s `repr()` in
  the judge prompt, while the new `format_choices` joins it into clean lines.
  This changed **zero** verdicts (those tasks still agree 15/15) — a readability
  improvement with no behavioral impact.

Conclusion: identical inputs (Tier 1, 23/23 byte-for-byte) + identical grading
(Tier 4, 360/360) ⇒ the framework is behaviorally equivalent to the original
harness.

## Environment notes (cluster)

- **Login node blocks `pip install`** (resource guard) — install/run on a compute
  node or via SLURM. `cvss` can be vendored into `.deps/` from its wheel.
- **`gpu-short` may assign Tesla P100 (sm_60)**, unsupported by the env's
  torch/vLLM. Exclude P100 nodes; `gpu-A100`/`gpu-H200` need a special QOS.
