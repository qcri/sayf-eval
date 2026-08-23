"""Results-record tests: pipeline-config embedding, no item-text leakage, save,
and the security gates on Hub push (offline; HfApi mocked)."""

import json
import sys
import types

from sayf_eval.model import GenParams
from sayf_eval.results import build_record, push_details, push_scores, save_record


SUMMARY = {
    "mcq": {"accuracy": 0.5, "correct": 1, "total": 2, "skipped": 0},
    "vsp": {"accuracy": 0.0, "correct": 0, "total": 2, "skipped": 0, "mad": 1.2},
}

_HF = "RISys-Lab/Benchmarks_CyberSec_CTI-Bench"
TASK_SOURCES = {
    "mcq": {
        "type": "hf_dataset",
        "dataset_name": "CTI-Bench MCQ",
        "hf_repo": _HF,
        "subset": "cti-mcq",
        "split": "test",
    },
    "vsp": {
        "type": "hf_dataset",
        "dataset_name": "CTI-Bench VSP (CVSS)",
        "hf_repo": _HF,
        "subset": "cti-vsp",
        "split": "test",
    },
}


def _record():
    return build_record(
        SUMMARY,
        model="openai/gpt-4o",
        judge="anthropic/claude-sonnet-4",
        gen_params=GenParams(),
        model_base_url=None,
        max_tokens_override=4096,
        answer_stop=["\n"],
        task_sources=TASK_SOURCES,
    )


def test_record_embeds_pipeline_config_and_scores():
    r = _record().to_dict()
    assert r["model"]["name"] == "openai/gpt-4o" and r["model"]["provider"] == "openai"
    assert r["judge"]["provider"] == "anthropic"
    assert r["results"] == SUMMARY
    assert sorted(r["tasks"]) == ["mcq", "vsp"]
    # the differentiator: decoding + scoring policy travels with the scores
    p = r["pipeline"]
    assert p["temperature"] == 0.0 and p["top_p"] == 1.0 and p["seed"] == 42
    assert p["max_tokens_override"] == 4096 and p["answer_stop"] == ["\n"]
    assert "denominator" in p["denominator_policy"].lower()
    assert r["schema_version"] and r["sayf_eval_version"] and r["created_at"]


def test_record_embeds_task_sources():
    # The record is self-describing: each task carries its declared provenance.
    r = _record().to_dict()
    assert r["task_sources"]["mcq"]["type"] == "hf_dataset"
    assert r["task_sources"]["mcq"]["hf_repo"].startswith("RISys-Lab/")
    assert r["task_sources"]["vsp"]["subset"] == "cti-vsp"


def test_record_embeds_judge_prompt_templates():
    # The record carries the actual extract+verdict prompt scaffolding (placeholders
    # only), keyed by task, so exporters record the real judge prompt not a blurb.
    import sayf_eval.tasks  # noqa: F401 — register
    from sayf_eval.cli import _judge_prompt_templates
    from sayf_eval.registry import get_task

    tpls = _judge_prompt_templates([get_task("mcq"), get_task("secure_kcv")])
    r = build_record(
        SUMMARY,
        model="openai/gpt-4o",
        judge="anthropic/claude-sonnet-4",
        gen_params=GenParams(),
        task_sources=TASK_SOURCES,
        judge_prompt_templates=tpls,
    ).to_dict()
    assert r["judge_prompt_templates"]["mcq"].startswith("You are a strict evaluator")
    # secure_kcv uses its True/False rule, not the A–E MCQ one
    assert "T (true) or F (false)" in r["judge_prompt_templates"]["secure_kcv"]
    # placeholders only — no item text
    assert "{question}" in r["judge_prompt_templates"]["mcq"]


def test_scores_record_carries_no_item_text():
    # Hard invariant: the scores artifact must never contain per-sample item text.
    d = _record().to_dict()
    # judge_prompt_templates are static prompt scaffolding (placeholders only, no
    # item text); exclude that field so its schema tokens don't false-positive.
    d.pop("judge_prompt_templates", None)
    blob = json.dumps(d).lower()
    for forbidden in ("model_response", "extracted_answer", "ground_truth", "question"):
        assert forbidden not in blob


def test_save_record_writes_per_model_path(tmp_path):
    path = save_record(_record(), str(tmp_path))
    assert path.endswith(".json") and "openai__gpt-4o" in path
    on_disk = json.loads(open(path).read())
    assert on_disk["results"]["mcq"]["accuracy"] == 0.5


class _FakeApi:
    def __init__(self, token=None):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
        self.created.append({"repo_id": repo_id, "private": private})

    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, **kw):
        self.uploaded.append({"repo_id": repo_id, "path": path_in_repo})


def _install_fake_hub(monkeypatch):
    mod = types.ModuleType("huggingface_hub")
    fake = _FakeApi()
    mod.HfApi = lambda token=None: fake
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return fake


def test_push_scores_private_by_default(monkeypatch):
    fake = _install_fake_hub(monkeypatch)
    repo = push_scores(_record(), "myorg", public=False)
    assert repo == "myorg/openai__gpt-4o__sayf-eval"
    assert fake.created[0]["private"] is True  # private by default
    assert fake.uploaded[0]["path"].startswith("results/")


def test_push_scores_public_when_requested(monkeypatch):
    fake = _install_fake_hub(monkeypatch)
    push_scores(_record(), "myorg", public=True)
    assert fake.created[0]["private"] is False


def test_push_details_is_always_private(monkeypatch, tmp_path):
    fake = _install_fake_hub(monkeypatch)
    (tmp_path / "mcq_detailed.jsonl").write_text('{"index":0,"is_correct":true}\n')
    repo = push_details(str(tmp_path), "myorg", "openai/gpt-4o")
    assert repo == "myorg/details_openai__gpt-4o"
    assert fake.created[0]["private"] is True  # details NEVER public
    assert any(u["path"] == "mcq_detailed.jsonl" for u in fake.uploaded)
