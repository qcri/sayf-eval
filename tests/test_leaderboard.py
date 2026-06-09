"""Leaderboard (Level 2) emission tests — benchmark eval.yaml + per-model
.eval_results YAML, schema correctness, pipeline-config in notes, and gated push.
Offline; HfApi mocked."""

import sys
import types

import yaml

from sayf_eval.leaderboard import (
    EVALUATION_FRAMEWORK,
    build_eval_results,
    build_eval_yaml,
    push_benchmark,
)


RECORD = {
    "model": {"name": "openai/gpt-4o", "provider": "openai"},
    "judge": {"name": "anthropic/claude-sonnet-4", "provider": "anthropic"},
    "pipeline": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
    "results": {
        "mcq": {"accuracy": 0.6667, "correct": 2, "total": 3, "skipped": 0},
        "vsp": {"accuracy": 0.5, "correct": 1, "total": 2, "skipped": 0, "mad": 1.2},
    },
    "created_at": "2026-06-09T12:00:00+00:00",
}


def test_eval_yaml_schema():
    spec = yaml.safe_load(build_eval_yaml("Sayf-Eval", "cyber benchmark", ["mcq", "vsp"]))
    assert spec["name"] == "Sayf-Eval"
    assert spec["evaluation_framework"] == EVALUATION_FRAMEWORK
    assert [t["id"] for t in spec["tasks"]] == ["mcq", "vsp"]  # sorted, id-only


def test_eval_results_schema_and_percentage():
    entries = yaml.safe_load(build_eval_results(RECORD, "qcri/sayf-eval"))
    assert isinstance(entries, list) and len(entries) == 2
    e = {x["dataset"]["task_id"]: x for x in entries}
    assert e["mcq"]["dataset"]["id"] == "qcri/sayf-eval"
    assert e["mcq"]["value"] == 66.67  # accuracy ×100, rounded
    assert e["vsp"]["value"] == 50.0
    assert e["mcq"]["date"] == "2026-06-09"  # from created_at


def test_eval_results_fraction_mode():
    entries = yaml.safe_load(build_eval_results(RECORD, "qcri/sayf-eval", as_percentage=False))
    assert {x["dataset"]["task_id"]: x["value"] for x in entries}["mcq"] == 0.6667


def test_notes_carry_pipeline_config():
    entries = yaml.safe_load(build_eval_results(RECORD, "qcri/sayf-eval"))
    notes = entries[0]["notes"]
    # the leaderboard entry must never show a score without the pipeline behind it
    for token in ("judge=anthropic/claude-sonnet-4", "temp=0", "seed=42", "all-attempted"):
        assert token in notes


def test_eval_results_task_subset():
    entries = yaml.safe_load(build_eval_results(RECORD, "qcri/sayf-eval", task_ids=["mcq"]))
    assert [x["dataset"]["task_id"] for x in entries] == ["mcq"]


class _FakeApi:
    def __init__(self, token=None):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
        self.created.append({"repo_id": repo_id, "private": private})

    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, **kw):
        self.uploaded.append(path_in_repo)


def test_push_benchmark_private_by_default(monkeypatch):
    mod = types.ModuleType("huggingface_hub")
    fake = _FakeApi()
    mod.HfApi = lambda token=None: fake
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    repo = push_benchmark("qcri/sayf-eval", "name: X\n", public=False)
    assert repo == "qcri/sayf-eval"
    assert fake.created[0]["private"] is True
    assert "eval.yaml" in fake.uploaded
