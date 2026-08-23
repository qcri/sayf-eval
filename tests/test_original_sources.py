"""Offline tests for the original-source loaders (CTI-Bench AI4Sec, SECURE
aiforsec TSV, CyberMetric JSON, CTI-TAA gold). HTTP / HF access is mocked."""

import json

import pytest

from sayf_eval import datasets as ds


def test_cti_ai4sec_mcq_renders_from_question_and_options(monkeypatch):
    import datasets as hf

    rows = [
        {
            "URL": "u",
            "Question": "Q?",
            "Option A": "a",
            "Option B": "b",
            "Option C": "c",
            "Option D": "d",
            "Prompt": "full-prompt",
            "GT": "B",
        }
    ]
    monkeypatch.setattr(hf, "load_dataset", lambda *a, **k: rows)
    s = ds.make_cti_ai4sec_loader("AI4Sec/cti-bench", "cti-mcq", "mcq")()
    assert len(s) == 1
    assert s[0].target == "B"
    assert s[0].choices == ["A. a", "B. b", "C. c", "D. d"]
    assert "Question: Q?" in s[0].prompt and s[0].prompt.endswith("Answer:")


def test_cti_ai4sec_structured_uses_prompt_column(monkeypatch):
    import datasets as hf

    rows = [{"URL": "u", "Description": "d", "Prompt": "the rcm prompt", "GT": "CWE-79"}]
    monkeypatch.setattr(hf, "load_dataset", lambda *a, **k: rows)
    s = ds.make_cti_ai4sec_loader("AI4Sec/cti-bench", "cti-rcm", "rcm")()
    assert s[0].prompt == "the rcm prompt"
    assert s[0].target == "CWE-79"
    assert s[0].choices is None


def test_secure_orig_maet_options_and_gold(monkeypatch):
    tsv = (
        "URL\tPrompt\tQuestion\tOption A\tOption B\tOption C\tOption D\tCorrect Answer\nu\tPfull\tQ?\ta\tb\tc\td\tA\n"
    )
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: tsv.encode())
    s = ds.make_secure_orig_loader("MAET", "secure")()
    assert s[0].prompt == "Pfull"
    assert s[0].target == "A"
    assert s[0].choices == ["A. a", "B. b", "C. c", "D. d"]


def test_secure_orig_kcv_gold_no_options(monkeypatch):
    tsv = "URL\tPrompt\tQuestion\tCorrect Answer\nu\tPfull\tIs it true?\tF\n"
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: tsv.encode())
    s = ds.make_secure_orig_loader("KCV", "secure_kcv")()
    assert s[0].prompt == "Pfull" and s[0].target == "F" and s[0].choices is None


def test_cybermetric_dict_schema(monkeypatch):
    payload = json.dumps(
        {"questions": [{"question": "Q?", "answers": {"A": "a", "B": "b", "C": "c", "D": "d"}, "solution": "A"}]}
    )
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    s = ds.load_cybermetric()
    assert len(s) == 1 and s[0].target == "A" and "Q?" in s[0].prompt


def test_cybermetric_toplevel_list_schema(monkeypatch):
    payload = json.dumps([{"question": "Q2?", "answers": {"A": "a", "B": "b"}, "solution": "B"}])
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    s = ds.load_cybermetric()
    assert len(s) == 1 and s[0].target == "B"


def test_cti_taa_gold_index_aligned_and_nonempty(monkeypatch):
    data = "URL\tText\tPrompt\nu1\tt1\tP1\nu2\tt2\tP2\n"
    gold = "GT\tChatGPT\nSideCopy\tX\nMUSTANG PANDA\tX\n"

    def fake(url, **k):
        return (data if url.endswith("data/cti-taa.tsv") else gold).encode()

    monkeypatch.setattr(ds, "_http_get", fake)
    s = ds.load_cti_taa()
    assert [x.target for x in s] == ["SideCopy", "MUSTANG PANDA"]
    assert all(x.target for x in s)  # never empty gold


def test_cti_taa_raises_on_length_mismatch(monkeypatch):
    data = "URL\tText\tPrompt\nu1\tt1\tP1\nu2\tt2\tP2\n"
    gold = "GT\nSideCopy\n"  # 1 gold vs 2 prompts

    def fake(url, **k):
        return (data if url.endswith("data/cti-taa.tsv") else gold).encode()

    monkeypatch.setattr(ds, "_http_get", fake)
    with pytest.raises(ValueError):
        ds.load_cti_taa()
