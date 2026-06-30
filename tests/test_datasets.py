"""Dataset loader helper tests (pure, offline) + registry coverage."""

from sayf_eval.datasets import _choices_to_list, _normalize_gt, _render_mcq


def test_choices_dict_skips_empty_and_orders_letters():
    assert _choices_to_list({"A": "x", "B": "y", "C": "z", "D": "w", "E": ""}) == ["A. x", "B. y", "C. z", "D. w"]


def test_choices_list_letter_prefixed():
    assert _choices_to_list(["x", "y", "z", "w"]) == ["A. x", "B. y", "C. z", "D. w"]


def test_choices_empty_is_none():
    assert _choices_to_list(None) is None
    assert _choices_to_list({}) is None


def test_normalize_gt_priority_and_int_to_letter():
    assert _normalize_gt({"GT": "CWE-79", "answer": 0}) == "CWE-79"  # GT wins
    assert _normalize_gt({"answer": 2}) == "C"  # int -> letter
    assert _normalize_gt({"label": " b "}) == "b"
    assert _normalize_gt({}) == ""


def test_render_mcq_shape():
    out = _render_mcq("INSTR", "Q?", ["A. x", "B. y"])
    assert out.startswith("INSTR") and out.endswith("Answer:")
    assert "A. x\nB. y" in out


def test_registry_covers_full_suite():
    import sayf_eval.tasks  # noqa: F401 — registers
    from sayf_eval.judge_prompts import create_judge_prompt
    from sayf_eval.registry import available_tasks, get_task

    tasks = available_tasks()
    assert len(tasks) >= 24
    for n in tasks:
        t = get_task(n)
        # Code-gen tasks (scorer_kind="code") are scored by static analysis, not
        # the judge, so they have no judge prompt — skip them.
        if getattr(t, "scorer_kind", "") == "code":
            continue
        # every judged task_type must resolve a judge prompt
        create_judge_prompt(t.task_type, "q", "a", "g", {"choices": "A. x"})
