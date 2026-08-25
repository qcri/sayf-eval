"""Command-line entrypoint: ``sayf-eval {run-inference,run-judge,run}``.

The run subcommands build the model-under-test and/or judge as the *same*
:class:`~sayf_eval.model.Model` type — local vLLM is reached by passing
``--model openai/<served-name> --base-url http://localhost:8000/v1``.

Tasks are registered by importing ``sayf_eval.tasks`` (see :func:`_import_tasks`),
which happens once at startup in :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from sayf_eval.model import GenParams, Model
from sayf_eval.pipeline import RunConfig, run_inference, run_judge, run_tasks
from sayf_eval.scorer import JudgeScorer


def _import_tasks() -> None:
    """Import the task package so registrations populate the registry."""
    try:
        import sayf_eval.tasks  # noqa: F401
    except ModuleNotFoundError:
        # Defensive: if the tasks package can't be imported, the registry stays
        # empty and task resolution raises a clear error downstream.
        pass


def _build_model(args) -> Model:
    return Model(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
    )


def _build_scorer(args) -> JudgeScorer:
    judge = Model(
        model=args.judge,
        base_url=args.judge_base_url or args.base_url,
        api_key=args.judge_api_key or args.api_key,
        concurrency=args.concurrency,
    )
    return JudgeScorer(judge, GenParams(max_tokens=args.judge_max_tokens))


def _resolve_tasks(names: list[str]):
    from sayf_eval.registry import available_tasks, get_task

    if not available_tasks():
        sys.exit("No tasks registered — sayf_eval.tasks failed to import.")
    return [get_task(n) for n in names]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tasks", nargs="+", required=True, help="Task names (registry keys).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--answer-stop",
        nargs="*",
        default=None,
        help="Post-think stop sequence(s) applied to the answer before judging.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the per-task generation budget (e.g. scale up for "
        "thinking models that emit reasoning before the answer).",
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="LiteLLM model string, e.g. openai/gpt-4o.")
    p.add_argument("--base-url", default=None, help="Endpoint override (set for local vLLM).")
    p.add_argument("--api-key", default=None)


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge", required=True, help="LiteLLM model string for the judge.")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--judge-max-tokens", type=int, default=512)


def _add_push_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results-org", default=None, help="HF org/user to push results to.")
    p.add_argument(
        "--push-scores",
        action="store_true",
        help="Push the scores record (metrics + pipeline config, no item text) to the Hub.",
    )
    p.add_argument(
        "--push-details",
        action="store_true",
        help="Also push per-sample details (prompt/gold/response) — ALWAYS to a private repo.",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Make the scores repo public (details stay private regardless).",
    )
    p.add_argument("--hf-token", default=None, help="HF token (else uses $HF_TOKEN).")


def _judge_prompt_templates(tasks) -> dict:
    """Render each task's judge prompt as a template (placeholders, no item text).

    Lets the results record carry the actual extract+verdict prompt scaffolding
    so downstream exporters record the real judge prompt rather than a description.
    """
    from sayf_eval.judge_prompts import create_judge_prompt

    templates: dict[str, str] = {}
    for t in tasks:
        try:
            templates[t.name] = create_judge_prompt(
                t.task_type,
                "{question}",
                "{model_answer}",
                "{ground_truth}",
                {"choices": "{choices}"},
            )
        except ValueError:
            continue  # unknown task_type — skip rather than fabricate a prompt
    return templates


def _export_results(args, summary: dict, tasks) -> None:
    """Build + save the canonical scores record, then push if requested."""
    from sayf_eval.results import build_record, push_details, push_scores, save_record

    record = build_record(
        summary,
        model=args.model,
        judge=args.judge,
        gen_params=GenParams(),
        model_base_url=args.base_url,
        judge_base_url=args.judge_base_url or args.base_url,
        max_tokens_override=args.max_tokens,
        answer_stop=args.answer_stop,
        task_sources={t.name: (t.source or {}) for t in tasks},
        judge_prompt_templates=_judge_prompt_templates(tasks),
    )
    path = save_record(record, args.output_dir)
    print(f"results record → {path}")

    if args.push_scores or args.push_details:
        if not args.results_org:
            sys.exit("--results-org is required to push to the Hub.")
    if args.push_scores:
        repo = push_scores(record, args.results_org, public=args.public, token=args.hf_token)
        print(f"scores → {'public' if args.public else 'private'} dataset {repo}")
    if args.push_details:
        repo = push_details(args.output_dir, args.results_org, args.model, token=args.hf_token)
        print(f"details → private dataset {repo}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sayf-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inf = sub.add_parser("run-inference", help="Collect model responses.")
    _add_common(p_inf)
    _add_model_args(p_inf)

    p_judge = sub.add_parser("run-judge", help="Judge already-collected responses.")
    _add_common(p_judge)
    _add_judge_args(p_judge)

    p_run = sub.add_parser("run", help="Inference + judge end to end.")
    _add_common(p_run)
    _add_model_args(p_run)
    _add_judge_args(p_run)
    _add_push_args(p_run)

    args = parser.parse_args(argv)
    _import_tasks()

    tasks = _resolve_tasks(args.tasks)
    cfg = RunConfig(
        max_samples=args.max_samples,
        answer_stop=args.answer_stop,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
    )

    if args.cmd == "run-inference":
        model = _build_model(args)
        for task in tasks:
            path = run_inference(task, model, args.output_dir, cfg)
            print(f"{task.name}: responses → {path}")
        return 0

    if args.cmd == "run-judge":
        scorer = _build_scorer(args)
        summary = {}
        for task in tasks:
            responses_path = os.path.join(args.output_dir, f"{task.name}_responses.jsonl")
            summary[task.name] = run_judge(task, responses_path, scorer, args.output_dir, cfg)
            print(f"{task.name}: {summary[task.name]}")
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return 0

    if args.cmd == "run":
        model = _build_model(args)
        scorer = _build_scorer(args)
        summary = run_tasks(tasks, model, scorer, args.output_dir, cfg)
        print(json.dumps(summary, indent=2))
        _export_results(args, summary, tasks)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
