"""Judge-prompt routing tests for secure_kcv (T/F/X vs A–E MCQ). No network."""

from sayf_eval.judge_prompts import _MCQ_TYPES, _RULES, create_judge_prompt


_MCQ_HINT = "A, B, C, D, or E"


def test_secure_kcv_resolves_to_tfx_rule():
    # secure_kcv must go through _RULES, not the MCQ path.
    assert "secure_kcv" in _RULES
    assert "secure_kcv" not in _MCQ_TYPES


def test_secure_kcv_prompt_uses_tfx_hint_not_mcq():
    p = create_judge_prompt(
        "secure_kcv",
        "Is the statement true?",
        "The claim is false.",
        "F",
    )
    # True/False verification hints are present ...
    assert "T (true)" in p
    assert "F (false)" in p
    # ... and it does NOT fall back to the A–E multiple-choice hint.
    assert _MCQ_HINT not in p


def test_secure_kcv_abstention_maps_to_none_not_x():
    # X is a valid gold label ("unknown"), so a no-answer response must fall back
    # to the NONE sentinel (always incorrect), not to X — otherwise unanswered
    # items would score correct whenever the gold answer is X.
    p = create_judge_prompt("secure_kcv", "Is the statement true?", "", "X")
    # X stays reserved for a deliberate "does not know" answer ...
    assert "X when the model" in p
    assert "does not know" in p
    # ... and truly-absent answers route to NONE, which is always incorrect.
    assert '"NONE"' in p
    assert "always INCORRECT" in p


def test_secure_kcv_registered_with_kcv_task_type():
    # Guards the actual production bug: the registry must bind secure_kcv to the
    # "secure_kcv" scoring family, not "secure". If this reverts, evaluation goes
    # back down the MCQ path even though the rule above still exists.
    import sayf_eval.tasks  # noqa: F401 — importing registers the full suite
    from sayf_eval.registry import get_task

    t = get_task("secure_kcv")
    assert t.task_type == "secure_kcv"
    assert t.scorer_kind == "secure_kcv"
    assert t.task_type not in _MCQ_TYPES


def test_secure_mcq_path_still_uses_ae_hint():
    # Regression guard: the MCQ SECURE path is unchanged and diverges from KCV,
    # so the two never collapse back into one routing.
    p = create_judge_prompt("secure", "Pick one option.", "The answer is B.", "B")
    assert _MCQ_HINT in p
    assert "T (true)" not in p
