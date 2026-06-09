"""Stdlib test runner — runs the same assertions as the pytest suite without
requiring pytest (the cluster login node blocks pip installs).

In a normal environment, prefer ``pytest tests/``. This runner mirrors those
tests so the core can be validated here. Mocks ``litellm`` in-process.
"""

import sys
import types
import traceback

# --- mock litellm before importing seceval.model ---------------------------

_calls = []
_next = {"fn": None}


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)


def _install_fake_litellm():
    mod = types.ModuleType("litellm")
    mod.drop_params = False
    _calls.clear()

    def completion(**kwargs):
        _calls.append(kwargs)
        return _next["fn"](**kwargs)

    mod.completion = completion
    sys.modules["litellm"] = mod
    _next["fn"] = lambda **kw: _Resp("A")


_install_fake_litellm()

from seceval.model import GenParams, Model           # noqa: E402
from seceval.scorer import parse_judge_response, strip_reasoning  # noqa: E402
from seceval.metrics import (                          # noqa: E402
    calculate_vsp_mad, compute_ate_metrics, score_corpus, set_prf1,
    split_id_set, parent_only,
)
from seceval.datasets import (                          # noqa: E402
    _choices_to_list, _normalize_gt, _render_mcq,
)

_results = []


def check(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
    except Exception:
        _results.append((name, False, traceback.format_exc()))


# --- model ------------------------------------------------------------------

def t_generate_ok():
    _next["fn"] = lambda **kw: _Resp("A")
    m = Model("openai/gpt-x")
    r = m.generate([{"role": "user", "content": "hi"}], GenParams(max_tokens=16))
    assert r.ok and r.text == "A", r
    assert r.usage["total_tokens"] == 3
    assert _calls[-1]["model"] == "openai/gpt-x"
    assert _calls[-1]["max_completion_tokens"] == 16
    assert _calls[-1]["temperature"] == 0.0


def t_empty_not_ok():
    _next["fn"] = lambda **kw: _Resp("")
    m = Model("openai/gpt-x", num_retries=1)
    r = m.generate([{"role": "user", "content": "hi"}], GenParams())
    assert (not r.ok) and r.text == ""


def t_content_filter_single_attempt():
    def boom(**kw):
        raise RuntimeError("blocked by Microsoft content management policy")
    _next["fn"] = boom
    _calls.clear()
    m = Model("openai/gpt-x", num_retries=5)
    r = m.generate([{"role": "user", "content": "x"}], GenParams())
    assert not r.ok
    assert len(_calls) == 1, len(_calls)


def t_base_url_routing():
    _next["fn"] = lambda **kw: _Resp("A")
    m = Model("openai/qwen", base_url="http://localhost:8000/v1", api_key="EMPTY")
    m.generate([{"role": "user", "content": "x"}], GenParams())
    assert _calls[-1]["base_url"] == "http://localhost:8000/v1"
    assert _calls[-1]["api_key"] == "EMPTY"


def t_batch_order():
    seq = iter(["A", "B", "C"])
    _next["fn"] = lambda **kw: _Resp(next(seq))
    m = Model("openai/gpt-x", concurrency=1)
    out = m.generate_batch([[{"role": "user", "content": str(i)}] for i in range(3)], GenParams())
    assert [r.text for r in out] == ["A", "B", "C"]


# --- scorer -----------------------------------------------------------------

def t_clean_json_correct():
    v = parse_judge_response('{"extracted_answer": "B", "verdict": "CORRECT", "justification": "ok"}', "mcq")
    assert v.is_correct and not v.skipped and v.extracted_answer == "B"


def t_fenced_json():
    v = parse_judge_response('```json\n{"extracted_answer": "A", "verdict": "INCORRECT", "justification": "n"}\n```', "mcq")
    assert (not v.is_correct) and not v.skipped and v.extracted_answer == "A"


def t_prose_around_json():
    v = parse_judge_response('Sure!\n{"extracted_answer": "T1059", "verdict": "CORRECT", "justification": "x"} done', "ate")
    assert v.is_correct and v.extracted_answer == "T1059"


def t_regex_fallback():
    raw = '{"extracted_answer": "CVSS:3.1/AV:N", "verdict": "CORRECT", "justification": "y",,,}'
    v = parse_judge_response(raw, "vsp")
    assert v.is_correct and v.extracted_answer == "CVSS:3.1/AV:N"


def t_empty_skipped():
    assert parse_judge_response("   ", "mcq").skipped


def t_error_skipped():
    assert parse_judge_response("ERROR: timeout", "mcq").skipped


def t_filter_skipped():
    assert parse_judge_response("blocked by content management policy", "mcq").skipped


def t_strip_think():
    assert strip_reasoning("<think>cot</think>\nAnswer: B") == "Answer: B"


def t_strip_think_stop():
    assert strip_reasoning("<think>n\nl</think>B\nextra", stop=["\n"]) == "B"


# --- metrics ----------------------------------------------------------------

def t_vsp_identical():
    v = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert calculate_vsp_mad(v, v) == 0.0


def t_vsp_prefix_norm():
    a = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    b = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert calculate_vsp_mad(a, b) == 0.0


def t_vsp_invalid_max():
    assert calculate_vsp_mad("nope", "bad") == 10.0


def t_set_exact():
    assert set_prf1({"A", "B"}, {"A", "B"})["f1"] == 1.0


def t_set_partial():
    r = set_prf1({"A"}, {"A", "B"})
    assert r["precision"] == 1.0 and r["recall"] == 0.5 and r["exact_match"] == 0


def t_parent_only():
    assert parent_only(split_id_set("T1059.001,T1059.003,T1027")) == {"T1059", "T1027"}


def t_ate_micro():
    rows = [
        {"extracted_answer": "T1059.001", "ground_truth": "T1059"},
        {"extracted_answer": "T1027", "ground_truth": "T1027,T1055"},
    ]
    m = compute_ate_metrics(rows, parent=True)
    assert m["tp_total"] == 2 and m["fn_total"] == 1 and m["fp_total"] == 0


def t_corpus_denominator():
    rows = [
        {"is_correct": True, "skipped": False},
        {"is_correct": False, "skipped": False},
        {"is_correct": False, "skipped": True},
    ]
    r = score_corpus("mcq", rows)
    assert r["correct"] == 1 and r["total"] == 2 and r["skipped"] == 1 and r["accuracy"] == 0.5


def t_corpus_vsp():
    rows = [{"is_correct": True, "skipped": False,
             "extracted_answer": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
             "ground_truth": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}]
    r = score_corpus("vsp", rows)
    assert r["mad"] == 0.0 and r["extraction_success"] == 1


# --- datasets (loader helpers) ---------------------------------------------

def t_choices_dict_skip_empty():
    assert _choices_to_list({"A": "x", "B": "y", "C": "z", "D": "w", "E": ""}) == [
        "A. x", "B. y", "C. z", "D. w"]


def t_choices_list():
    assert _choices_to_list(["x", "y"]) == ["A. x", "B. y"]


def t_choices_none():
    assert _choices_to_list(None) is None and _choices_to_list({}) is None


def t_normalize_gt():
    assert _normalize_gt({"GT": "CWE-79", "answer": 0}) == "CWE-79"
    assert _normalize_gt({"answer": 2}) == "C"
    assert _normalize_gt({"label": " b "}) == "b"
    assert _normalize_gt({}) == ""


def t_render_mcq():
    out = _render_mcq("INSTR", "Q?", ["A. x", "B. y"])
    assert out.startswith("INSTR") and out.endswith("Answer:") and "A. x\nB. y" in out


def t_registry_full_suite():
    import seceval.tasks  # noqa: F401
    from seceval.registry import available_tasks, get_task
    from seceval.judge_prompts import create_judge_prompt
    tasks = available_tasks()
    assert len(tasks) >= 24
    for n in tasks:
        t = get_task(n)
        create_judge_prompt(t.task_type, "q", "a", "g", {"choices": "A. x"})


for _name, _fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("t_")):
    check(_name, _fn)

_passed = sum(1 for _, ok, _ in _results if ok)
for name, ok, tb in _results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(tb)
print(f"\n{_passed}/{len(_results)} passed")
sys.exit(0 if _passed == len(_results) else 1)
