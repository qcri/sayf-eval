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
        # Tasks package not present yet (pre-Phase-2); registry stays empty.
        pass


def _build_model(args):
    """Build the model-under-test: an endpoint Model or in-process OfflineVLLMModel."""
    if getattr(args, "model_backend", "endpoint") == "offline-vllm":
        from sayf_eval.backends import OfflineVLLMModel

        return OfflineVLLMModel(
            model=args.model,
            tensor_parallel_size=args.model_tp_size,
            gpu_memory_utilization=args.model_gpu_mem_util,
            max_model_len=args.model_max_model_len,
            trust_remote_code=args.model_trust_remote_code,
        )
    return Model(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
    )


def _build_translator(args):
    """Build the translation layer, or ``None`` for the English baseline."""
    from sayf_eval.translate import build_translator

    kind = getattr(args, "translator", "none")
    if kind == "none":
        return None
    offline = {
        "tensor_parallel_size": args.translator_tp_size,
        "gpu_memory_utilization": args.translator_gpu_mem_util,
        "max_model_len": args.translator_max_model_len,
        "trust_remote_code": args.translator_trust_remote_code,
    }
    return build_translator(
        kind,
        target_lang=args.translator_lang,
        backend=args.translator_backend,
        model=args.translator_model,
        base_url=args.translator_base_url or getattr(args, "base_url", None),
        api_key=args.translator_api_key or getattr(args, "api_key", None),
        concurrency=args.concurrency,
        cache_dir=args.translator_cache_dir,
        write_cache_dir=args.translator_write_cache,
        model_aware=args.translator_model_aware,
        offline=offline,
    )


def _build_scorer(args) -> JudgeScorer | None:
    """Build the LLM judge, or ``None`` when no ``--judge`` is given (code-only runs)."""
    if not getattr(args, "judge", None):
        return None
    judge = Model(
        model=args.judge,
        base_url=args.judge_base_url or args.base_url,
        api_key=args.judge_api_key or args.api_key,
        concurrency=args.concurrency,
    )
    return JudgeScorer(judge, GenParams(max_tokens=args.judge_max_tokens))


def _build_code_scorer(args, tasks):
    """Build a static-analysis :class:`CodeScorer` if any task needs one, else ``None``."""
    if not any(getattr(t, "scorer_kind", "") == "code" for t in tasks):
        return None
    from sayf_eval.codescore import CodeScorer

    return CodeScorer(kind=getattr(args, "code_analyzer", "auto"))


def _require_judge_for_non_code(args, tasks) -> None:
    """Non-code tasks need an LLM judge; fail fast if one is missing."""
    non_code = [t.name for t in tasks if getattr(t, "scorer_kind", "") != "code"]
    if non_code and not getattr(args, "judge", None):
        sys.exit(f"--judge is required for non-code tasks: {', '.join(non_code)}")


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
    p.add_argument("--model", required=True, help="Model id/path (LiteLLM string for endpoint; checkpoint for offline-vllm).")
    p.add_argument(
        "--model-backend",
        choices=["endpoint", "offline-vllm"],
        default="endpoint",
        help="endpoint (LiteLLM/vLLM server via --base-url) or offline-vllm (in-process, server-less).",
    )
    p.add_argument("--base-url", default=None, help="Endpoint override (set for local vLLM).")
    p.add_argument("--api-key", default=None)
    # offline-vllm loading knobs (mirror the offline batched jobs)
    p.add_argument("--model-tp-size", type=int, default=1, help="offline-vllm tensor_parallel_size.")
    p.add_argument("--model-gpu-mem-util", type=float, default=0.90, help="offline-vllm gpu_memory_utilization.")
    p.add_argument("--model-max-model-len", type=int, default=None, help="offline-vllm max_model_len.")
    p.add_argument("--model-trust-remote-code", action="store_true", help="offline-vllm trust_remote_code.")


def _add_translator_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--translator",
        choices=["none", "llm", "cache"],
        default="none",
        help="Translation layer: none (English baseline), llm (translate at run "
        "time with --translator-model), or cache (read pre-translated prompts).",
    )
    p.add_argument(
        "--translator-lang",
        default="ar",
        choices=["ar", "en"],
        help="Target language: ar (EN->AR) or en (AR->EN translate-test).",
    )
    p.add_argument("--translator-model", default=None, help="Model id/path for the translator (llm).")
    p.add_argument(
        "--translator-backend",
        choices=["endpoint", "offline-vllm"],
        default="endpoint",
        help="endpoint (LiteLLM/vLLM server) or offline-vllm (in-process, server-less).",
    )
    p.add_argument("--translator-base-url", default=None, help="Translator endpoint (else reuses --base-url).")
    p.add_argument("--translator-api-key", default=None)
    # offline-vllm loading knobs (mirror the offline batched jobs)
    p.add_argument("--translator-tp-size", type=int, default=1, help="offline-vllm tensor_parallel_size.")
    p.add_argument("--translator-gpu-mem-util", type=float, default=0.90, help="offline-vllm gpu_memory_utilization.")
    p.add_argument("--translator-max-model-len", type=int, default=None, help="offline-vllm max_model_len.")
    p.add_argument(
        "--translator-trust-remote-code",
        action="store_true",
        help="offline-vllm trust_remote_code.",
    )
    p.add_argument(
        "--translator-cache-dir",
        default=None,
        help="Directory of <task>.jsonl pre-translated prompts (cache).",
    )
    p.add_argument(
        "--translator-write-cache",
        default=None,
        help="Write run-time translations to this dir for later reuse (llm).",
    )
    p.add_argument(
        "--translator-model-aware",
        action="store_true",
        help="Use the per-model MCQ translation prompts (Fanar/LLaMA/GPT-OSS).",
    )


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    # Optional: not needed for code-gen-only runs (those use --code-analyzer).
    p.add_argument("--judge", default=None, help="LiteLLM model string for the judge (omit for code-gen-only runs).")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-api-key", default=None)
    p.add_argument("--judge-max-tokens", type=int, default=512)
    p.add_argument(
        "--code-analyzer",
        choices=["auto", "bandit", "codeshield"],
        default="auto",
        help="Static analyzer for code-gen tasks (securityeval, cse_instruct, cse_autocomplete). "
        "auto=CodeShield if installed else Bandit; bandit=Python only; codeshield=multi-language.",
    )


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


def _export_results(args, summary: dict) -> None:
    """Build + save the canonical scores record, then push if requested."""
    from sayf_eval.results import build_record, push_details, push_scores, save_record

    record = build_record(
        summary,
        model=args.model,
        judge=args.judge or "",
        gen_params=GenParams(),
        model_base_url=args.base_url,
        judge_base_url=args.judge_base_url or args.base_url,
        max_tokens_override=args.max_tokens,
        answer_stop=args.answer_stop,
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


def _cmd_benchmark_spec(args) -> int:
    from sayf_eval.leaderboard import build_eval_yaml, push_benchmark
    from sayf_eval.registry import available_tasks

    task_ids = args.tasks or available_tasks()
    if not task_ids:
        sys.exit("No tasks registered.")
    yaml_str = build_eval_yaml(args.name, args.description, task_ids)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(yaml_str)
    print(f"benchmark eval.yaml → {args.out} ({len(task_ids)} tasks)")
    if args.push_to:
        repo = push_benchmark(args.push_to, yaml_str, public=args.public, token=args.token)
        print(f"benchmark → {'public' if args.public else 'private'} dataset {repo}")
    return 0


def _cmd_translate(args) -> int:
    """Pre-translate tasks to a cache dir — load one model in-process, batch, write.

    Mirrors the offline batched jobs: build the translator once (offline-vllm or an
    endpoint), translate every task's prompts, and write
    ``<output-dir>/<task>.jsonl`` for later reuse via ``--translator cache``. No
    model-under-test or judge is loaded, so a single GPU holds only the translator.
    """
    import json as _json

    from sayf_eval.translate import LLMTranslator

    tasks = _resolve_tasks(args.tasks)
    args.translator = "llm"  # this command is translation-only
    translator = _build_translator(args)
    if not isinstance(translator, LLMTranslator):
        sys.exit("translate: expected an LLM translator (set --translator-model).")

    os.makedirs(args.output_dir, exist_ok=True)
    for task in tasks:
        samples = task.load(args.max_samples)
        out = translator.translate_samples(task, samples)
        system_prompt = translator.system_prompt(task)
        path = os.path.join(args.output_dir, f"{task.name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in out:
                row = {"index": s.index, "prompt": s.prompt, "system_prompt": system_prompt}
                f.write(_json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{task.name}: {len(out)} translated → {path}")
    print(f"\nTranslation cache → {args.output_dir} (run with --translator cache --translator-cache-dir {args.output_dir})")
    return 0


def _cmd_eval_results(args) -> int:
    from sayf_eval.leaderboard import build_eval_results, submit_results_pr

    with open(args.results, encoding="utf-8") as f:
        record = json.load(f)
    yaml_str = build_eval_results(
        record,
        args.benchmark_id,
        task_ids=args.tasks,
        as_percentage=not args.no_percentage,
    )
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(yaml_str)
    print(f".eval_results → {args.out}")
    if args.submit_pr:
        url = submit_results_pr(args.submit_pr, yaml_str, token=args.token)
        print(f"community results PR → {url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sayf-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inf = sub.add_parser("run-inference", help="Collect model responses.")
    _add_common(p_inf)
    _add_model_args(p_inf)
    _add_translator_args(p_inf)

    p_judge = sub.add_parser("run-judge", help="Judge already-collected responses.")
    _add_common(p_judge)
    _add_judge_args(p_judge)

    p_run = sub.add_parser("run", help="Inference + judge end to end.")
    _add_common(p_run)
    _add_model_args(p_run)
    _add_translator_args(p_run)
    _add_judge_args(p_run)
    _add_push_args(p_run)

    # Standalone pre-translation pass (offline batched-job style): load one model,
    # translate tasks, write a reusable cache. No model-under-test / judge.
    p_tr = sub.add_parser("translate", help="Pre-translate tasks to a cache dir (no MUT/judge).")
    p_tr.add_argument("--tasks", nargs="+", required=True, help="Task names (registry keys).")
    p_tr.add_argument("--output-dir", required=True, help="Cache dir for <task>.jsonl outputs.")
    p_tr.add_argument("--max-samples", type=int, default=None)
    p_tr.add_argument("--concurrency", type=int, default=8)
    _add_translator_args(p_tr)

    # Leaderboard (Level 2 — HF Community-Evals). Opt-in; publishes scores.
    p_spec = sub.add_parser(
        "benchmark-spec", help="Emit the benchmark dataset eval.yaml (register sayf-eval as a HF benchmark)."
    )
    p_spec.add_argument("--name", default="Sayf-Eval — Cybersecurity LLM Evaluation")
    p_spec.add_argument(
        "--description",
        default="Model-agnostic cybersecurity LLM benchmark suite (CTI, vuln scoring, ICS/OT, tool proficiency, open-ended CTI extraction).",
    )
    p_spec.add_argument("--tasks", nargs="*", default=None, help="Task subset (default: all registered).")
    p_spec.add_argument("--out", default="eval.yaml")
    p_spec.add_argument("--push-to", default=None, help="Hub dataset id to create/update (e.g. qcri/sayf-eval).")
    p_spec.add_argument(
        "--public", action="store_true", help="Make the benchmark dataset public (deliberate disclosure)."
    )
    p_spec.add_argument("--token", default=None)

    p_er = sub.add_parser(
        "eval-results",
        help="Emit per-model .eval_results YAML from a results record (optionally PR it to a model repo).",
    )
    p_er.add_argument("--results", required=True, help="Path to a results record JSON (from a run).")
    p_er.add_argument(
        "--benchmark-id", required=True, help="Hub id of the registered benchmark dataset (e.g. qcri/sayf-eval)."
    )
    p_er.add_argument("--tasks", nargs="*", default=None)
    p_er.add_argument("--out", default=".eval_results/sayf-eval.yaml")
    p_er.add_argument(
        "--no-percentage", action="store_true", help="Emit accuracy as a 0–1 fraction instead of a percentage."
    )
    p_er.add_argument("--submit-pr", default=None, help="Model repo to open a community results PR against.")
    p_er.add_argument("--token", default=None)

    args = parser.parse_args(argv)
    _import_tasks()

    if args.cmd == "benchmark-spec":
        return _cmd_benchmark_spec(args)
    if args.cmd == "eval-results":
        return _cmd_eval_results(args)
    if args.cmd == "translate":
        return _cmd_translate(args)

    tasks = _resolve_tasks(args.tasks)
    cfg = RunConfig(
        max_samples=args.max_samples,
        answer_stop=args.answer_stop,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
    )

    if args.cmd == "run-inference":
        model = _build_model(args)
        translator = _build_translator(args)
        for task in tasks:
            path = run_inference(task, model, args.output_dir, cfg, translator)
            print(f"{task.name}: responses → {path}")
        return 0

    _require_judge_for_non_code(args, tasks)

    if args.cmd == "run-judge":
        scorer = _build_scorer(args)
        code_scorer = _build_code_scorer(args, tasks)
        summary = {}
        for task in tasks:
            responses_path = os.path.join(args.output_dir, f"{task.name}_responses.jsonl")
            summary[task.name] = run_judge(task, responses_path, scorer, args.output_dir, cfg, code_scorer)
            print(f"{task.name}: {summary[task.name]}")
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return 0

    if args.cmd == "run":
        model = _build_model(args)
        translator = _build_translator(args)
        scorer = _build_scorer(args)
        code_scorer = _build_code_scorer(args, tasks)
        summary = run_tasks(tasks, model, scorer, args.output_dir, cfg, translator, code_scorer)
        print(json.dumps(summary, indent=2))
        _export_results(args, summary)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
