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

from sayf_eval import metrics
from sayf_eval.model import GenParams, Model
from sayf_eval.scorer import JudgeScorer
from sayf_eval.task import Sample, Task
from sayf_eval.translate import Translator


@dataclass
class RunConfig:
    """Knobs for a run."""

    max_samples: int | None = None
    answer_stop: list[str] | None = None  # post-think stop applied before judging
    overwrite: bool = False
    max_tokens: int | None = None  # override per-task budget (e.g. scale up
    # for thinking models that emit reasoning)
    mcq_render: str = "default"  # "letter" re-renders MCQ prompts (EN-side parity
    # with --ar-render seedmini)


def _messages(task: Task, sample: Sample, system_prompt: str | None) -> list[dict]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
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
    translator: Translator | None = None,
) -> str:
    """Generate model responses for ``task``; return the responses JSONL path.

    When ``translator`` is given, the loaded samples and the task system prompt are
    rewritten into the translator's target language before inference. Ground truth
    and choices are untouched, so judging and metrics are unchanged.
    """
    config = config or RunConfig()
    out_path = os.path.join(output_dir, f"{task.name}_responses.jsonl")
    if os.path.exists(out_path) and not config.overwrite:
        return out_path

    samples = task.load(config.max_samples)
    if config.mcq_render == "letter":
        # EN-side parity with --ar-render seedmini: render MCQ prompts in the
        # Question:/A:/.. layout. Set only on the run that needs it (e.g. the EN
        # baseline); AR runs let the translator own rendering.
        from sayf_eval.gemma_translate import re_render_letter

        samples = re_render_letter(samples)
    system_prompt = task.system_prompt
    if translator is not None:
        samples = translator.translate_samples(task, samples)
        system_prompt = translator.system_prompt(task)
    params = GenParams(max_tokens=config.max_tokens or task.max_tokens)
    batch = [_messages(task, s, system_prompt) for s in samples]
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
    code_scorer=None,
) -> dict:
    """Judge collected responses; write detailed JSONL; return corpus scores.

    Code-generation tasks (``task.scorer_kind == "code"``) are routed to
    ``code_scorer`` (a :class:`~sayf_eval.codescore.CodeScorer`, static analysis)
    instead of the LLM ``scorer``; everything downstream is identical.
    """
    config = config or RunConfig()
    rows_in = _read_jsonl(responses_path)

    items = [
        {
            "task_type": task.task_type,
            "question": r.get("prompt", ""),
            "model_answer": r.get("model_response", ""),
            "target": r.get("ground_truth", ""),
            "choices": (r.get("metadata") or {}).get("choices"),
            "metadata": r.get("metadata") or {},
            "answer_stop": config.answer_stop,
        }
        for r in rows_in
    ]
    if task.scorer_kind == "code":
        if code_scorer is None:
            raise ValueError(
                f"Task {task.name!r} is a code-gen task (scorer_kind='code') but no code scorer was "
                "provided. Pass --code-analyzer (bandit|codeshield|auto)."
            )
        active = code_scorer
    else:
        active = scorer
    verdicts = active.score_batch(items)

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
    translator: Translator | None = None,
    code_scorer=None,
) -> dict:
    responses_path = run_inference(task, model, output_dir, config, translator)
    return run_judge(task, responses_path, scorer, output_dir, config, code_scorer)


def run_tasks(
    tasks: list[Task],
    model: Model,
    scorer: JudgeScorer,
    output_dir: str,
    config: RunConfig | None = None,
    translator: Translator | None = None,
    code_scorer=None,
) -> dict:
    """Run several tasks; write ``summary.json``; return the summary dict."""
    summary: dict[str, dict] = {}
    for task in tasks:
        summary[task.name] = run_task(task, model, scorer, output_dir, config, translator, code_scorer)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
