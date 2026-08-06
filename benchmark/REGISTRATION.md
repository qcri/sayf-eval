# HF benchmark registration steps

Two one-time actions make the leaderboard live.

## 1. Register the `sayf-eval` evaluation framework (PR to `huggingface.js`)

HF maintains the framework enum in
[`packages/tasks/src/eval.ts`](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/eval.ts)
(`EVALUATION_FRAMEWORKS`). Open a PR adding:

```typescript
"sayf-eval": {
  name: "sayf-eval",
  description: "Model-agnostic framework for evaluating LLMs on cybersecurity benchmarks (CTI, vulnerability scoring, ICS/OT, tool proficiency, open-ended CTI extraction).",
  url: "https://github.com/qcri/sayf-eval",
},
```

> Alternative without a framework PR: set `evaluation_framework: inspect-ai` in
> `eval.yaml` and supply `field_spec` / `solvers` / `scores`. We use our own
> identifier because sayf-eval is LiteLLM+judge based, not Inspect-AI.

## 2. Create + allow-list the benchmark dataset

```bash
# private first to review the rendered card/leaderboard, then make public
sayf-eval benchmark-spec --push-to qcri/sayf-eval                 # private
# ... review ...
sayf-eval benchmark-spec --push-to qcri/sayf-eval --public        # publish
```

Then upload the dataset card and request allow-listing (registration is beta):

- Upload `DATASET_CARD.md` as the dataset repo `README.md`.
- Contact HF to add `qcri/sayf-eval` to the benchmark allow-list.

## 3. Seed the leaderboard with results

For each evaluated model, attach scores via a community PR:

```bash
sayf-eval eval-results \
  --results out/<model>/results/*/results_*.json \
  --benchmark-id qcri/sayf-eval \
  --submit-pr <model_repo>
```

PRs show as "community-provided" until the model author merges; authors can close
disputed scores.
