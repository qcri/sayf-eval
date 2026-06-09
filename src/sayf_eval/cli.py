"""Command-line entrypoint: ``seceval {run-inference,run-judge,run}``.

All three subcommands build the model-under-test and/or judge as the *same*
:class:`~sayf_eval.model.Model` type — local vLLM is reached by passing
``--model openai/<served-name> --base-url http://localhost:8000/v1``.

Tasks must be registered (importing ``sayf_eval.tasks`` triggers registration once
the MVP loaders land in Phase 2).
"""

from __future__ import annotations

import argparse
import json
import sys

from sayf_eval.model import GenParams, Model
from sayf_eval.pipeline import RunConfig, run_inference, run_judge, run_tasks
from sayf_eval.scorer import JudgeScorer


def _import_tasks() -> None:
    """Import the task package so registrations populate the registry."""
    try:
        import sayf_eval.tasks  # noqa: F401
    except ModuleNotFoundError:
        # Tasks package not present yet (pre-Phase-2); registry stays empty.
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
        sys.exit("No tasks registered. (MVP task loaders arrive in Phase 2 — see PLAN.md.)")
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
            import os

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
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
