"""Judge-prompt routing tests for secure_kcv (T/F/X vs A–E MCQ). No network."""

from sayf_eval.judge_prompts import _MCQ_TYPES, _RULES, create_judge_prompt


_TFX_HINT = "T (true), F (false), or X (unknown)"
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
    assert _TFX_HINT in p  # uses the True/False/X extraction hint
    assert _MCQ_HINT not in p  # and does NOT fall back to the A–E MCQ hint


def test_secure_mcq_path_still_uses_ae_hint():
    # Regression guard: the MCQ SECURE path is unchanged and diverges from KCV,
    # so the two never collapse back into one routing.
    p = create_judge_prompt("secure", "Pick one option.", "The answer is B.", "B")
    assert _MCQ_HINT in p
    assert _TFX_HINT not in p
