"""Leaderboard Level 2 — HuggingFace Community-Evals emission.

Two artifacts, per the Hub spec (https://huggingface.co/docs/hub/eval-results):

  1. A **benchmark dataset** ``eval.yaml`` (repo root) that registers sayf-eval as
     a benchmark and declares its tasks as sub-leaderboards.
  2. Per-model ``.eval_results/*.yaml`` files — one entry per task
     ``{dataset:{id,task_id}, value, date, source, notes}`` — that surface on the
     model card and aggregate into the benchmark's leaderboard. The ``notes`` field
     carries the pipeline-config summary, so the public leaderboard never shows a
     score without the pipeline behind it.

Security note: this path **publishes** scores. It is entirely opt-in and writes
files locally by default; pushing to the Hub / opening PRs are separate explicit
calls. Creating the benchmark dataset is private unless ``public=True`` — going
public with a security-eval leaderboard is a deliberate disclosure decision.

YAML + huggingface_hub are pulled via the ``[hub]`` extra.
"""

from __future__ import annotations

import os


# Identifier sayf-eval reports under. To register as a real HF benchmark this must
# be added to HF's evaluation-framework enum (huggingface.js) and the dataset
# allow-listed (beta) — see push_benchmark()'s printed guidance.
EVALUATION_FRAMEWORK = "sayf-eval"


def _yaml():
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Leaderboard emission needs PyYAML — pip install 'sayf-eval[hub]'") from e
    return yaml


def _dump(obj) -> str:
    return _yaml().safe_dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _pipeline_notes(pipeline: dict, judge_name: str) -> str:
    """One-line eval-setup summary for the leaderboard ``notes`` field."""
    return (
        f"judge={judge_name}; temp={pipeline.get('temperature')}; "
        f"top_p={pipeline.get('top_p')}; seed={pipeline.get('seed')}; "
        f"denom=all-attempted(unparseable=incorrect); <think>-stripped; "
        f"scoring=llm-judge"
    )


# -- benchmark dataset eval.yaml --------------------------------------------


def build_eval_yaml(
    name: str,
    description: str,
    task_ids: list[str],
    framework: str = EVALUATION_FRAMEWORK,
) -> str:
    """Build the benchmark dataset's ``eval.yaml`` (one sub-leaderboard per task)."""
    spec = {
        "name": name,
        "description": description,
        "evaluation_framework": framework,
        "tasks": [{"id": tid} for tid in sorted(task_ids)],
    }
    return _dump(spec)


# -- per-model .eval_results/*.yaml -----------------------------------------


def build_eval_results(
    record: dict,
    benchmark_id: str,
    *,
    task_ids: list[str] | None = None,
    as_percentage: bool = True,
    source_url: str | None = None,
    source_name: str = "sayf-eval",
) -> str:
    """Build the per-model ``.eval_results`` YAML list from a results record.

    ``record`` is a :class:`~sayf_eval.results.ResultsRecord` dict. One entry per
    task; ``value`` is accuracy (×100 by default). ``benchmark_id`` is the Hub id
    of the registered benchmark dataset (e.g. ``"qcri/sayf-eval"``).
    """
    results = record.get("results", {})
    pipeline = record.get("pipeline", {})
    judge_name = (record.get("judge") or {}).get("name", "")
    date = (record.get("created_at") or "")[:10] or None
    notes = _pipeline_notes(pipeline, judge_name)
    tasks = sorted(task_ids) if task_ids else sorted(results)

    entries = []
    for task in tasks:
        metrics = results.get(task)
        if not metrics or "accuracy" not in metrics:
            continue
        if as_percentage:
            value = round(metrics["accuracy"] * 100, 2)
        else:
            value = round(metrics["accuracy"], 4)
        entry = {
            "dataset": {"id": benchmark_id, "task_id": task},
            "value": value,
            "notes": notes,
        }
        if date:
            entry["date"] = date
        if source_url:
            entry["source"] = {"url": source_url, "name": source_name}
        entries.append(entry)
    return _dump(entries)


# -- optional Hub push / PR (gated, opt-in) ---------------------------------


def _hf_api(token: str | None):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Hub push needs huggingface_hub — pip install 'sayf-eval[hub]'") from e
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def push_benchmark(
    repo_id: str,
    eval_yaml: str,
    *,
    dataset_card: str | None = None,
    public: bool = False,
    token: str | None = None,
) -> str:
    """Create/update the benchmark dataset repo with ``eval.yaml`` (+ optional card).

    Private unless ``public=True``. A public benchmark leaderboard for security
    tasks is a deliberate disclosure decision. Registration also requires HF to
    add ``evaluation_framework`` to their enum and allow-list the dataset (beta).
    """
    api = _hf_api(token)
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    api.upload_file(
        path_or_fileobj=eval_yaml.encode("utf-8"),
        path_in_repo="eval.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="sayf-eval benchmark spec",
    )
    if dataset_card:
        api.upload_file(
            path_or_fileobj=dataset_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="sayf-eval benchmark card",
        )
    if not public:
        print(f"NOTE: {repo_id} is PRIVATE — it will not aggregate a public leaderboard until made public.")
    print(
        "NOTE: HF benchmark registration is beta — the evaluation_framework "
        f"'{EVALUATION_FRAMEWORK}' must be added to HF's enum and the dataset allow-listed."
    )
    return repo_id


def submit_results_pr(
    model_repo: str,
    eval_results_yaml: str,
    *,
    filename: str = "sayf-eval.yaml",
    token: str | None = None,
) -> str:
    """Open a PR on a model repo adding ``.eval_results/<filename>``.

    Returns the PR URL. Submitting as a PR (vs. direct commit) is the community
    path — it shows as "community-provided" until the model author merges.
    """
    from huggingface_hub import CommitOperationAdd

    api = _hf_api(token)
    commit = api.create_commit(
        repo_id=model_repo,
        operations=[
            CommitOperationAdd(
                path_in_repo=f".eval_results/{filename}",
                path_or_fileobj=eval_results_yaml.encode("utf-8"),
            )
        ],
        commit_message="Add sayf-eval results",
        create_pr=True,
    )
    return getattr(commit, "pr_url", "") or str(commit)
