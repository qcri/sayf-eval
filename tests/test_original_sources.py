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


def test_cti_ai4sec_mcq_skips_row_with_empty_question(monkeypatch):
    import datasets as hf

    rows = [
        {"URL": "u0", "Question": "", "Option A": "a", "Option B": "b",
         "Option C": "c", "Option D": "d", "GT": "A"},  # no question -> skip
        {"URL": "u1", "Question": "Real?", "Option A": "a", "Option B": "b",
         "Option C": "c", "Option D": "d", "GT": "B"},
    ]
    monkeypatch.setattr(hf, "load_dataset", lambda *a, **k: rows)
    s = ds.make_cti_ai4sec_loader("AI4Sec/cti-bench", "cti-mcq", "mcq")()
    assert [x.target for x in s] == ["B"]
    assert "Question: Real?" in s[0].prompt


def test_cti_ai4sec_mcq_skips_whitespace_only_options(monkeypatch):
    import datasets as hf

    # Whitespace-only options are truthy but strip to empty -> choices would be
    # None; the row must be skipped rather than crash _render_mcq.
    rows = [
        {"URL": "u0", "Question": "Q?", "Option A": "   ", "Option B": "\t", "GT": "A"},
        {
            "URL": "u1",
            "Question": "Real?",
            "Option A": "a",
            "Option B": "b",
            "Option C": "c",
            "Option D": "d",
            "GT": "B",
        },
    ]
    monkeypatch.setattr(hf, "load_dataset", lambda *a, **k: rows)
    s = ds.make_cti_ai4sec_loader("AI4Sec/cti-bench", "cti-mcq", "mcq")()
    assert [x.target for x in s] == ["B"]


def test_cti_ai4sec_mcq_skips_partial_option_set(monkeypatch):
    import datasets as hf

    # CTI-Bench MCQ is a strict 4-option (A-D) question; a row missing an option
    # (e.g. only A/B) would render choices inconsistent with the "A, B, C, D"
    # instruction, so it must be skipped rather than rendered.
    rows = [
        {"URL": "u0", "Question": "Partial?", "Option A": "a", "Option B": "b", "GT": "A"},
        {
            "URL": "u1",
            "Question": "Full?",
            "Option A": "a",
            "Option B": "b",
            "Option C": "c",
            "Option D": "d",
            "GT": "C",
        },
    ]
    monkeypatch.setattr(hf, "load_dataset", lambda *a, **k: rows)
    s = ds.make_cti_ai4sec_loader("AI4Sec/cti-bench", "cti-mcq", "mcq")()
    assert [x.target for x in s] == ["C"]  # partial A/B row dropped, full A-D kept


def test_cti_ai4sec_mcq_restricts_to_ad_and_none_safe_gold(monkeypatch):
    import datasets as hf

    # A stray Option E must be dropped (instruction is A-D); GT=None must become
    # "" rather than the literal string "None".
    rows = [
        {
            "URL": "u",
            "Question": "Q?",
            "Option A": "a",
            "Option B": "b",
            "Option C": "c",
            "Option D": "d",
            "Option E": "e",
            "GT": None,
        }
    ]
    monkeypatch.setattr(hf, "load_dataset", lambda *a, **k: rows)
    s = ds.make_cti_ai4sec_loader("AI4Sec/cti-bench", "cti-mcq", "mcq")()
    assert s[0].choices == ["A. a", "B. b", "C. c", "D. d"]
    assert "E. e" not in s[0].prompt
    assert s[0].target == ""


def test_secure_orig_maet_options_and_gold(monkeypatch):
    tsv = (
        "URL\tPrompt\tQuestion\tOption A\tOption B\tOption C\tOption D\tCorrect Answer\nu\tPfull\tQ?\ta\tb\tc\td\tA\n"
    )
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: tsv.encode())
    s = ds.make_secure_orig_loader("MAET", "secure")()
    assert s[0].prompt == "Pfull"
    assert s[0].target == "A"
    assert s[0].choices == ["A. a", "B. b", "C. c", "D. d"]


def test_secure_orig_maet_restricts_to_ad(monkeypatch):
    # A stray Option E must not leak into choices (SECURE MAET is A-D).
    tsv = (
        "URL\tPrompt\tQuestion\tOption A\tOption B\tOption C\tOption D\tOption E\tCorrect Answer\n"
        "u\tPfull\tQ?\ta\tb\tc\td\te\tA\n"
    )
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: tsv.encode())
    s = ds.make_secure_orig_loader("MAET", "secure")()
    assert s[0].choices == ["A. a", "B. b", "C. c", "D. d"]


def test_secure_orig_maet_skips_partial_option_set(monkeypatch):
    # SECURE MAET is a strict 4-option MCQ: a row with only A/B (some but not all
    # of A-D) would render choices that disagree with the task's A-D instruction,
    # so it must be dropped. A full A-D row on the same TSV is the control.
    tsv = (
        "URL\tPrompt\tQuestion\tOption A\tOption B\tOption C\tOption D\tCorrect Answer\n"
        "u0\tPpartial\tQ?\ta\tb\t\t\tA\n"  # only A/B populated -> skip
        "u1\tPfull\tQ?\ta\tb\tc\td\tB\n"  # full A-D -> kept
    )
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: tsv.encode())
    s = ds.make_secure_orig_loader("MAET", "secure")()
    assert [x.target for x in s] == ["B"]
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
    payload = json.dumps([{"question": "Q2?", "answers": {"A": "a", "B": "b", "C": "c", "D": "d"}, "solution": "B"}])
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    s = ds.load_cybermetric()
    assert len(s) == 1 and s[0].target == "B"


def test_cybermetric_skips_incomplete_options(monkeypatch):
    # Only three options -> cannot render a consistent "A, B, C, or D" prompt; skip.
    payload = json.dumps([{"question": "Q?", "answers": {"A": "a", "B": "b", "C": "c"}, "solution": "A"}])
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    assert ds.load_cybermetric() == []


def test_cybermetric_list_answers_null_option_skips_row(monkeypatch):
    # answers given as a LIST with a null element: str(None) would become "None"
    # and survive the A-D filter, rendering "B) None". The row must be skipped.
    payload = json.dumps(
        [
            {"question": "Bad?", "answers": ["a", None, "c", "d"], "solution": "A"},
            {"question": "Ok?", "answers": ["a", "b", "c", "d"], "solution": "B"},
        ]
    )
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    s = ds.load_cybermetric()
    assert [x.target for x in s] == ["B"]
    assert "None" not in s[0].prompt


def test_cybermetric_raises_on_non_list_questions(monkeypatch):
    payload = json.dumps({"questions": {"not": "a list"}})
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    with pytest.raises(ValueError):
        ds.load_cybermetric()


def test_cybermetric_option_order_matches_choices(monkeypatch):
    # answers deliberately supplied out of A->D order; prompt + choices must both
    # come out A->D so the letters line up.
    payload = json.dumps([{"question": "Q?", "answers": {"C": "c", "A": "a", "D": "d", "B": "b"}, "solution": "A"}])
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    s = ds.load_cybermetric()
    assert s[0].choices == ["A. a", "B. b", "C. c", "D. d"]
    assert "A) a, B) b, C) c, D) d" in s[0].prompt


def test_cybermetric_drops_stray_e_option(monkeypatch):
    # The prompt instructs "A, B, C, or D" only; a stray E must not leak into
    # the rendered options or choices.
    payload = json.dumps(
        [{"question": "Q?", "answers": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}, "solution": "A"}]
    )
    monkeypatch.setattr(ds, "_http_get", lambda url, **k: payload.encode())
    s = ds.load_cybermetric()
    assert s[0].choices == ["A. a", "B. b", "C. c", "D. d"]
    assert "E)" not in s[0].prompt and "E. e" not in " ".join(s[0].choices)


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


def test_cti_taa_raises_on_null_gold_cell(monkeypatch):
    # A short gold row leaves the trailing GT column as None (DictReader restval),
    # which must fail fast rather than stringify to "None" and slip past the
    # emptiness check (the exact bypass the None-safe coercion closes).
    data = "URL\tText\tPrompt\nu1\tt1\tP1\nu2\tt2\tP2\n"
    gold = "ChatGPT\tGT\nX\tSideCopy\nY\n"  # row 2 has no GT field -> None

    def fake(url, **k):
        return (data if url.endswith("data/cti-taa.tsv") else gold).encode()

    monkeypatch.setattr(ds, "_http_get", fake)
    with pytest.raises(ValueError):
        ds.load_cti_taa()
