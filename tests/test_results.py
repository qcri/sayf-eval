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


def _record():
    return build_record(
        SUMMARY,
        model="openai/gpt-4o",
        judge="anthropic/claude-sonnet-4",
        gen_params=GenParams(),
        model_base_url=None,
        max_tokens_override=4096,
        answer_stop=["\n"],
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


def test_scores_record_carries_no_item_text():
    # Hard invariant: the scores artifact must never contain prompt/answer text.
    blob = json.dumps(_record().to_dict()).lower()
    for forbidden in ("prompt", "model_response", "extracted_answer", "ground_truth"):
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
