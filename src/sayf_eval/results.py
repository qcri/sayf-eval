"""Results record + optional Hub export (the "leaderboard" Level 1).

A sayf-eval result is only meaningful alongside the pipeline that produced it
(the project's thesis: benchmark scores are pipeline-dependent). So the canonical
record embeds the full pipeline configuration next to the scores — making entries
comparable by construction rather than bare numbers.

Security posture (cybersecurity benchmark):
  - The **scores record** carries per-task metrics + pipeline config only — never
    prompt / gold-answer / model-response text. Safe to publish.
  - **Details** (prompt/gold/response JSONL) are the sensitive asset (benchmark
    leakage + dual-use) and are pushed only on an explicit opt-in, always private.
  - Nothing is pushed unless asked; pushed repos are private unless ``public=True``.

The Hub layout mirrors lighteval's: ``{results_org}/{model_org}__{model_name}``
for scores and ``{results_org}/details_{model}`` for details. ``huggingface_hub``
is an optional dependency (``pip install 'sayf-eval[hub]'``).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from sayf_eval import __version__
from sayf_eval.model import GenParams


SCHEMA_VERSION = "1.0"

# Fixed, documented standardized choices (see README "Standardized pipeline
# choices"). Recorded so a score is never reported without the policy behind it.
_DENOMINATOR_POLICY = (
    "accuracy = correct / total over all attempted items; unparseable/empty "
    "answers count as incorrect; only judge-API failures are excluded (skipped) "
    "from both numerator and denominator"
)
_THINK_HANDLING = "strip <think>...</think> before judging, then apply stop sequence to the answer only"
_SCORING = "llm-as-judge: single call performs extraction + CORRECT/INCORRECT verdict"


def _sanitize(model: str) -> str:
    """Turn a LiteLLM model string into a Hub-safe ``org__name`` segment."""
    return model.replace("/", "__").replace(":", "_").strip("_")


@dataclass
class ModelInfo:
    name: str  # LiteLLM model string, e.g. "openai/gpt-4o"
    provider: str | None = None  # prefix (openai, anthropic, azure, hosted_vllm)
    base_url: str | None = None  # set for local / self-hosted endpoints

    @staticmethod
    def from_model(model: str, base_url: str | None = None) -> ModelInfo:
        provider = model.split("/", 1)[0] if "/" in model else None
        return ModelInfo(name=model, provider=provider, base_url=base_url)


@dataclass
class ResultsRecord:
    """Canonical, pipeline-config-embedded results record (the scores artifact)."""

    model: ModelInfo
    judge: ModelInfo
    pipeline: dict  # the decoding/scoring config that produced the scores
    results: dict  # {task: {accuracy, correct, total, skipped, ...}}
    tasks: list[str]
    # {task: source_data dict} — declared dataset provenance per task (see
    # Task.source). Makes the record self-describing so downstream exporters need
    # no knowledge of sayf-eval internals. Carries no prompt/gold/response text.
    task_sources: dict = field(default_factory=dict)
    sayf_eval_version: str = __version__
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def build_record(
    summary: dict,
    model: str,
    judge: str,
    gen_params: GenParams,
    *,
    model_base_url: str | None = None,
    judge_base_url: str | None = None,
    max_tokens_override: int | None = None,
    answer_stop: list[str] | None = None,
    task_sources: dict[str, dict] | None = None,
) -> ResultsRecord:
    """Assemble the scores record from a run summary + the pipeline configuration.

    ``summary`` is the ``run_tasks`` output: ``{task: {accuracy, correct, total,
    skipped, ...}}`` — aggregates only, no per-item text. ``task_sources`` maps
    each task to its declared dataset provenance (``Task.source``); it is stored
    verbatim so the record is self-describing.
    """
    pipeline = {
        "temperature": gen_params.temperature,
        "top_p": gen_params.top_p,
        "seed": gen_params.seed,
        "max_tokens": ("override" if max_tokens_override else "per-task-calibrated"),
        "max_tokens_override": max_tokens_override,
        "answer_stop": answer_stop,
        "think_handling": _THINK_HANDLING,
        "scoring": _SCORING,
        "denominator_policy": _DENOMINATOR_POLICY,
    }
    return ResultsRecord(
        model=ModelInfo.from_model(model, model_base_url),
        judge=ModelInfo.from_model(judge, judge_base_url),
        pipeline=pipeline,
        results=summary,
        tasks=sorted(summary.keys()),
        task_sources=task_sources or {},
    )


def save_record(record: ResultsRecord, output_dir: str) -> str:
    """Write the scores record to ``<output_dir>/results/<model>/results_<ts>.json``."""
    model_dir = os.path.join(output_dir, "results", _sanitize(record.model.name))
    os.makedirs(model_dir, exist_ok=True)
    stamp = record.created_at.replace(":", "-")
    path = os.path.join(model_dir, f"results_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
    return path


# -- Hub export (optional; requires the [hub] extra) ------------------------


def _hf_api(token: str | None):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Hub export needs huggingface_hub — install with: pip install 'sayf-eval[hub]'") from e
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def push_scores(
    record: ResultsRecord,
    results_org: str,
    *,
    public: bool = False,
    token: str | None = None,
) -> str:
    """Push the scores record (metrics + config, no item text) to a Hub dataset.

    Repo: ``{results_org}/{sanitized_model}__sayf-eval``. Private unless
    ``public=True``. Returns the repo id.
    """
    api = _hf_api(token)
    repo_id = f"{results_org}/{_sanitize(record.model.name)}__sayf-eval"
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    stamp = record.created_at.replace(":", "-")
    payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")
    api.upload_file(
        path_or_fileobj=payload,
        path_in_repo=f"results/results_{stamp}.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"sayf-eval scores for {record.model.name}",
    )
    return repo_id


def push_details(
    detailed_dir: str,
    results_org: str,
    model: str,
    *,
    token: str | None = None,
) -> str:
    """Push per-sample details (prompt/gold/response JSONL) to a **private** dataset.

    These contain benchmark items, so the repo is always private regardless of any
    public flag. Repo: ``{results_org}/details_{sanitized_model}``.
    """
    api = _hf_api(token)
    repo_id = f"{results_org}/details_{_sanitize(model)}"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    files = [f for f in os.listdir(detailed_dir) if f.endswith("_detailed.jsonl")]
    if not files:
        raise FileNotFoundError(f"no *_detailed.jsonl found in {detailed_dir}")
    for fn in files:
        api.upload_file(
            path_or_fileobj=os.path.join(detailed_dir, fn),
            path_in_repo=fn,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"sayf-eval details for {model}",
        )
    return repo_id
