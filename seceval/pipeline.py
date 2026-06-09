"""Pipeline — wire Task → Model → Scorer → aggregation.

Three steps, mirroring the original harness but model-agnostic:

1. :func:`run_inference` — render each sample's messages, generate with the
   model-under-test, write ``<task>_responses.jsonl``.
2. :func:`run_judge` — judge collected responses, write
   ``<task>_detailed.jsonl`` and per-task corpus scores.
3. :func:`run_task` / :func:`run_tasks` — do both end to end and write
   ``summary.json``.

Output schemas are kept compatible with the original analysis modules:
responses carry ``index/prompt/ground_truth/model_response/metadata``; detailed
rows carry the judge verdict fields.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from seceval import metrics
from seceval.model import GenParams, Model
from seceval.scorer import JudgeScorer
from seceval.task import Sample, Task


@dataclass
class RunConfig:
    """Knobs for a run."""

    max_samples: int | None = None
    answer_stop: list[str] | None = None   # post-think stop applied before judging
    overwrite: bool = False
    max_tokens: int | None = None          # override per-task budget (e.g. scale up
                                           # for thinking models that emit reasoning)


def _messages(task: Task, sample: Sample) -> list[dict]:
    msgs: list[dict] = []
    if task.system_prompt:
        msgs.append({"role": "system", "content": task.system_prompt})
    msgs.append({"role": "user", "content": sample.prompt})
    return msgs


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# -- step 1: inference -------------------------------------------------------

def run_inference(
    task: Task,
    model: Model,
    output_dir: str,
    config: RunConfig | None = None,
) -> str:
    """Generate model responses for ``task``; return the responses JSONL path."""
    config = config or RunConfig()
    out_path = os.path.join(output_dir, f"{task.name}_responses.jsonl")
    if os.path.exists(out_path) and not config.overwrite:
        return out_path

    samples = task.load(config.max_samples)
    params = GenParams(max_tokens=config.max_tokens or task.max_tokens)
    batch = [_messages(task, s) for s in samples]
    responses = model.generate_batch(batch, params)

    rows = [
        {
            "index": s.index,
            "prompt": s.prompt,
            "ground_truth": s.target,
            "model_response": r.text,
            "metadata": {**s.metadata, **({"choices": s.choices} if s.choices else {})},
            "ok": r.ok,
        }
        for s, r in zip(samples, responses)
    ]
    _write_jsonl(out_path, rows)
    return out_path


# -- step 2: judge -----------------------------------------------------------

def run_judge(
    task: Task,
    responses_path: str,
    scorer: JudgeScorer,
    output_dir: str,
    config: RunConfig | None = None,
) -> dict:
    """Judge collected responses; write detailed JSONL; return corpus scores."""
    config = config or RunConfig()
    rows_in = _read_jsonl(responses_path)

    items = [
        {
            "task_type": task.task_type,
            "question": r.get("prompt", ""),
            "model_answer": r.get("model_response", ""),
            "target": r.get("ground_truth", ""),
            "choices": (r.get("metadata") or {}).get("choices"),
            "answer_stop": config.answer_stop,
        }
        for r in rows_in
    ]
    verdicts = scorer.score_batch(items)

    detailed: list[dict] = []
    for r, v in zip(rows_in, verdicts):
        row = {
            "index": r.get("index"),
            "question": r.get("prompt", ""),
            "ground_truth": r.get("ground_truth", ""),
            "model_response": r.get("model_response", ""),
            "extracted_answer": v.extracted_answer,
            "judge_response": v.judge_response,
            "judge_justification": v.justification,
            "is_correct": v.is_correct,
            "skipped": v.skipped,
        }
        detailed.append(row)

    detailed_path = os.path.join(output_dir, f"{task.name}_detailed.jsonl")
    _write_jsonl(detailed_path, detailed)

    return metrics.score_corpus(task.task_type, detailed)


# -- step 3: end to end ------------------------------------------------------

def run_task(
    task: Task,
    model: Model,
    scorer: JudgeScorer,
    output_dir: str,
    config: RunConfig | None = None,
) -> dict:
    responses_path = run_inference(task, model, output_dir, config)
    return run_judge(task, responses_path, scorer, output_dir, config)


def run_tasks(
    tasks: list[Task],
    model: Model,
    scorer: JudgeScorer,
    output_dir: str,
    config: RunConfig | None = None,
) -> dict:
    """Run several tasks; write ``summary.json``; return the summary dict."""
    summary: dict[str, dict] = {}
    for task in tasks:
        summary[task.name] = run_task(task, model, scorer, output_dir, config)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
