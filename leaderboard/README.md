# sayf-eval leaderboard

Ten cybersecurity LLMs evaluated across **24 sub-tasks** from 9 benchmark
families (CTI-Bench, AthenaBench, SECURE, RedSage-MCQ, CyberMetric, MMLU-CS,
SecBench, SecEval, SEvenLLM). Every cell is scored by a **single `gpt-5.4` judge
run** under sayf-eval's unified extract-and-verdict prompt (temperature 0, top_p
1, seed 42). Models are ordered by mean strict-verdict accuracy across all
populated cells.

Each cell aggregates the default judge run with the restored validation-split
items; `secure_kcv` is re-scored from the extracted answer because the default
judge was given an A–E format hint that rejected otherwise-valid T/F verdicts.

## Data layout

Each model has one standard sayf-eval **results record** (schema `1.1`) under
[`results/<model>/`](results/), identical in shape to what `sayf-eval run`
writes — scores + full pipeline config + per-task dataset provenance + the judge
prompt templates, and **no per-sample question/answer/response text** (aggregate
only, per the project's dual-use posture). [`leaderboard.json`](leaderboard.json)
is the ranked index over those records.

## Regenerate this table

The table is rendered from the committed records only (no private data needed):

```bash
python leaderboard/render_table.py leaderboard
```

<!-- BEGIN GENERATED TABLE (leaderboard/render_table.py) -->
### Ranked summary

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

_Judge: `gpt-5.4` · single extract-and-verdict run · 10 models × 24 sub-tasks · compiled 2026-08-24._

### Full results

| Task | Sonnet-4.6 | GPT-5.4 | Gemma-4-31B | Qwen3.6-35B | Primus-Nemo-70B | RedSage-8B | Llama-3.3-70B | GPT-oss-20B | Found-Sec-8B | Primus-Merged |
|---|---|---|---|---|---|---|---|---|---|---|
| _CTI-Bench_ |  |  |  |  |  |  |  |  |  |  |
| MCQ <sup>Acc</sup> | **84.1** | 78.5 | 74.4 | 73.0 | 69.2 | 65.2 | 65.6 | 69.3 | 56.4 | 45.0 |
| RCM <sup>Acc</sup> | 75.4 | 74.2 | 70.5 | 71.8 | 65.3 | **75.7** | 63.1 | 65.9 | 69.4 | 67.5 |
| VSP <sup>MAD↓</sup> | **0.81** | 1.02 | 0.93 | 1.34 | 1.34 | 1.42 | 1.49 | 1.95 | 1.39 | 1.91 |
| ATE <sup>Acc</sup> | 6.7 | 5.0 | **8.3** | 3.3 | 1.7 | 0.0 | 1.7 | 3.3 | 1.7 | 0.0 |
| TAA <sup>Acc</sup> | **82.0** | 70.0 | 50.0 | 20.0 | 40.0 | 30.0 | 38.0 | 26.0 | 20.0 | 14.0 |
| _AthenaBench_ |  |  |  |  |  |  |  |  |  |  |
| CKT <sup>Acc</sup> | **92.8** | 91.0 | 86.1 | 84.9 | 81.3 | 79.2 | 82.2 | 81.1 | 78.5 | 76.6 |
| RMS <sup>F1</sup> | **59.6** | 41.4 | 21.5 | 3.4 | 14.9 | 23.7 | 10.9 | 2.3 | 24.8 | 8.4 |
| TAA <sup>Acc</sup> | **42.0** | 30.0 | 22.0 | 16.0 | 19.0 | 19.0 | 18.0 | 11.0 | 21.0 | 19.0 |
| ATE <sup>Acc</sup> | **79.2** | 66.4 | 49.4 | 49.6 | 53.2 | 51.0 | 29.4 | 26.8 | 38.0 | 33.6 |
| RCM <sup>Acc</sup> | **73.6** | 71.5 | 64.8 | 65.7 | 57.8 | 68.3 | 61.0 | 58.2 | 60.5 | 55.7 |
| VSP <sup>MAD-norm</sup> | **88.7** | 85.7 | 87.8 | 58.9 | 74.6 | 72.0 | 71.6 | 75.7 | 65.7 | 72.3 |
| _SECURE_ |  |  |  |  |  |  |  |  |  |  |
| MAET <sup>Acc</sup> | **94.1** | 93.1 | 92.3 | 90.2 | 91.0 | 89.6 | 86.7 | 87.3 | 84.6 | 78.5 |
| CWET <sup>Acc</sup> | **95.0** | **95.0** | 92.3 | 92.0 | 94.1 | 91.5 | 90.0 | 88.9 | 83.4 | 78.4 |
| KCV <sup>Acc</sup> | 88.4 | 88.4 | 84.8 | **89.5** | 86.9 | 81.5 | 87.6 | 88.2 | 81.1 | 82.2 |
| _RedSage-MCQ_ |  |  |  |  |  |  |  |  |  |  |
| FW <sup>Acc</sup> | **90.2** | 88.1 | 86.2 | 84.6 | 84.6 | 84.4 | 83.0 | 81.1 | 77.4 | 73.4 |
| GEN <sup>Acc</sup> | **90.7** | 90.2 | 87.1 | 86.7 | 86.8 | 83.7 | 85.8 | 82.1 | 74.5 | 74.9 |
| Skills <sup>Acc</sup> | **93.6** | 93.2 | 91.4 | 91.6 | 89.5 | 88.8 | 89.4 | 90.2 | 81.1 | 80.7 |
| CLI <sup>Acc</sup> | **93.6** | 92.7 | 88.8 | 90.1 | 87.1 | 86.6 | 86.9 | 89.7 | 75.2 | 75.9 |
| Kali <sup>Acc</sup> | **87.2** | 86.4 | 81.2 | 83.3 | 80.3 | 80.5 | 80.3 | 80.5 | 68.7 | 70.4 |
| _CyberMetric_ |  |  |  |  |  |  |  |  |  |  |
| CyberMetric <sup>Acc</sup> | **96.2** | 96.0 | 95.2 | 95.6 | 93.4 | 90.2 | 93.0 | 92.0 | 85.2 | 86.0 |
| _MMLU-CS_ |  |  |  |  |  |  |  |  |  |  |
| MMLU-CS <sup>Acc</sup> | **90.0** | 87.0 | **90.0** | 88.0 | 76.0 | 82.0 | 81.0 | 86.0 | 76.0 | 74.0 |
| _SecBench_ |  |  |  |  |  |  |  |  |  |  |
| SecBench <sup>Acc</sup> | **89.7** | 87.7 | 86.7 | 89.1 | 84.8 | 80.2 | 82.1 | 81.6 | 70.9 | 66.3 |
| _SecEval_ |  |  |  |  |  |  |  |  |  |  |
| SecEval <sup>Acc</sup> | 71.8 | **82.0** | 78.3 | 74.2 | 72.8 | 74.2 | 69.7 | 72.6 | 57.0 | 61.7 |
| _SEvenLLM_ |  |  |  |  |  |  |  |  |  |  |
| SEvenLLM <sup>Acc</sup> | **92.8** | 89.8 | 86.9 | 82.3 | 89.8 | 87.5 | 85.4 | 85.5 | 81.2 | 77.8 |
| **Average** <sup>Acc</sup> | **76.0** | 73.4 | 69.5 | 65.0 | 64.7 | 64.1 | 62.8 | 61.9 | 57.4 | 54.7 |

<sup>Acc</sup> strict-verdict accuracy (%) · <sup>MAD↓</sup> CVSS mean-abs-deviation (lower better) · <sup>MAD-norm</sup> max(0,1−MAD/7.7)×100 · <sup>F1</sup> macro-F1 over extracted IDs (%). **Bold** = best per task.

**Column legend (short → model id):** Sonnet-4.6 = `claude-sonnet-4-6` · GPT-5.4 = `gpt-5.4` · Gemma-4-31B = `gemma-4-31B-it` · Qwen3.6-35B = `Qwen/Qwen3.6-35B-A3B` · Primus-Nemo-70B = `Llama-Primus-Nemotron-70B-Instruct` · RedSage-8B = `RISys-Lab/RedSage-Qwen3-8B-DPO` · Llama-3.3-70B = `Llama-3.3-70B-Instruct` · GPT-oss-20B = `openai/gpt-oss-20b` · Found-Sec-8B = `fdtn-ai/Foundation-Sec-8B-Instruct` · Primus-Merged = `trendmicro-ailab/Llama-Primus-Merged`
<!-- END GENERATED TABLE -->
