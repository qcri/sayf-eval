"""Gemma-backed Arabic MCQ translation tests (pure, offline — no model/network)."""

import json

from sayf_eval.gemma_translate import (
    GemmaArabicTranslator,
    GemmaFieldStore,
    mcq_fields,
    re_render_letter,
    substitute_fields,
)
from sayf_eval.task import Sample, Task


_EN_Q = "Which port does HTTPS use?"
_EN_CH = {"A": "21", "B": "80", "C": "443", "D": "22"}
_AR_Q = "أي منفذ يستخدمه HTTPS؟"
_AR_CH = {"A": "٢١", "B": "٨٠", "C": "٤٤٣", "D": "٢٢"}


def _sample():
    opts = ", ".join(f"{k}) {v}" for k, v in _EN_CH.items())
    prompt = f"Question: {_EN_Q}\nOptions: {opts}\n\nChoose the correct answer. 'ANSWER: X'"
    return Sample(
        index=0,
        prompt=prompt,
        target="C",
        choices=[f"{k}: {v}" for k, v in _EN_CH.items()],
        metadata={"task_type": "mcq"},
    )


def _task():
    return Task(name="t", task_type="mcq", loader=lambda *a: [], system_prompt="EN sys", max_tokens=64)


def _store(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps(
            {
                "question": _EN_Q,
                "answers": _EN_CH,
                "solution": "C",
                "translated_question": _AR_Q,
                "translated_choices": _AR_CH,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return GemmaFieldStore({"t": str(p)})


def test_mcq_fields_parses_question_and_choices():
    q, ch = mcq_fields(_sample())
    assert q == _EN_Q and ch == _EN_CH


def test_store_lookup_is_content_keyed():
    import pathlib
    import tempfile

    store = _store(pathlib.Path(tempfile.mkdtemp()))
    hit = store.lookup("t", "  which PORT does https use?  ")  # case/space-insensitive
    assert hit and hit["ar_q"] == _AR_Q and hit["ar_choices"]["C"] == "٤٤٣"
    assert store.lookup("t", "no such question") is None


def test_substitute_fields_keeps_wrapper_swaps_content():
    out = substitute_fields(_sample().prompt, _EN_Q, _AR_Q, _EN_CH, _AR_CH)
    assert "Question:" in out and "ANSWER: X" in out  # English wrapper intact
    assert _AR_Q in out and "٤٤٣" in out  # Arabic substituted
    assert _EN_Q not in out  # English question gone


def test_seedmini_render_uses_letter_and_arabic_system(tmp_path):
    t = GemmaArabicTranslator("seedmini", store=_store(tmp_path))
    s = t.translate_samples(_task(), [_sample()])[0]
    assert s.prompt.startswith("Question:\n") and "A: ٢١" in s.prompt
    assert "Reply ONLY with the letter" in s.prompt
    assert t.system_prompt(_task()).startswith("أنت خبير")  # SYS_AR
    assert s.target == "C"  # gold untouched


def test_harness_render_keeps_english_system_and_wrapper(tmp_path):
    t = GemmaArabicTranslator("harness", store=_store(tmp_path))
    s = t.translate_samples(_task(), [_sample()])[0]
    assert "Options:" in s.prompt and _AR_Q in s.prompt
    assert t.system_prompt(_task()) == "EN sys"  # English system kept


def test_store_miss_falls_back_to_english_then_live(tmp_path):
    miss = Sample(
        index=1,
        prompt="Question: brand new?\nOptions: A) x, B) y\n\nq",
        target="A",
        choices=["A: x", "B: y"],
        metadata={"task_type": "mcq"},
    )
    # no live -> identity (English kept), counted as a miss
    t = GemmaArabicTranslator("harness", store=_store(tmp_path))
    assert t.translate_samples(_task(), [miss])[0].prompt == miss.prompt
    assert t._misses == 1
    # with live -> used
    t2 = GemmaArabicTranslator(
        "harness", store=_store(tmp_path), live=lambda q, ch: ("ARQ", {k: "ع" + v for k, v in ch.items()})
    )
    out = t2.translate_samples(_task(), [miss])[0]
    assert t2._live_used == 1 and "ARQ" in out.prompt


def test_re_render_letter_changes_layout_not_gold():
    s = _sample()
    r = re_render_letter([s])[0]
    assert r.prompt.startswith("Question:\n") and "A: 21" in r.prompt
    assert r.target == s.target and r.index == s.index
