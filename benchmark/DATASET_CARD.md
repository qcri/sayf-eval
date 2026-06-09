---
license: mit
tags:
  - benchmark
  - cybersecurity
  - llm-evaluation
language:
  - en
---

# Sayf-Eval — Cybersecurity LLM Evaluation

A model-agnostic benchmark suite for evaluating LLMs on cybersecurity tasks,
produced by [sayf-eval](https://github.com/qcri/sayf-eval). It aggregates **25
sub-tasks across 9 benchmark families** into per-task leaderboards.

## What it measures

| Family | Tasks |
|--------|-------|
| CTI-Bench | `mcq`, `rcm`, `vsp`, `ate`, `cti_taa` |
| AthenaBench | `ckt`, `rms`, `taa`, `athena_ate`, `athena_rcm`, `athena_vsp` |
| SECURE (ICS/OT) | `secure_maet`, `secure_cwet`, `secure_kcv` |
| RedSage | `redsage_frameworks`, `redsage_generals`, `redsage_skills`, `redsage_cli`, `redsage_kali` |
| SecEval · CyberMetric · SecBench · MMLU-CS · CISSP | MCQ knowledge |
| SEvenLLM | `sevenllm` (open-ended structured CTI extraction) |

## How scores are produced (and why the pipeline travels with them)

Benchmark scores are **pipeline-dependent**. Every sayf-eval result carries its
full pipeline configuration in the result's `notes` (judge model, decoding params,
denominator policy, `<think>` handling), so a number on this leaderboard is never
reported without the pipeline that produced it. Standardized choices: temperature
0 / top_p 1 / fixed seed; per-task token budgets; `<think>` stripped before
judging; **accuracy = correct / total over all attempted items** (unparseable
answers count as incorrect; only judge-API failures are excluded).

## Submitting results

Run sayf-eval and open a community PR with your `.eval_results`:

```bash
sayf-eval run --tasks mcq … --model <m> --judge <j> --output-dir out/<m>
sayf-eval eval-results --results out/<m>/results/*/results_*.json \
  --benchmark-id qcri/sayf-eval --submit-pr <your_model_repo>
```

## Responsible use

This benchmark evaluates security-relevant capabilities. Per-sample **details
(prompts / answers) are not published** to avoid benchmark leakage and dual-use
exposure; only aggregate scores and the pipeline configuration are shared.

## License

MIT. See the [sayf-eval repository](https://github.com/qcri/sayf-eval).
