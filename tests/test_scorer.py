"""Scorer parsing + reasoning-strip tests (no network)."""

from seceval.scorer import parse_judge_response, strip_reasoning


def test_clean_json_correct():
    raw = '{"extracted_answer": "B", "verdict": "CORRECT", "justification": "ok"}'
    v = parse_judge_response(raw, "mcq")
    assert v.is_correct and not v.skipped and v.extracted_answer == "B"


def test_fenced_json_incorrect():
    raw = '```json\n{"extracted_answer": "A", "verdict": "INCORRECT", "justification": "no"}\n```'
    v = parse_judge_response(raw, "mcq")
    assert not v.is_correct and not v.skipped and v.extracted_answer == "A"


def test_prose_around_json():
    raw = 'Sure!\n{"extracted_answer": "T1059", "verdict": "CORRECT", "justification": "x"} done'
    v = parse_judge_response(raw, "ate")
    assert v.is_correct and v.extracted_answer == "T1059"


def test_regex_fallback_on_malformed_json():
    # Trailing junk makes json.loads fail; regex fallback recovers the keys.
    raw = '{"extracted_answer": "CVSS:3.1/AV:N", "verdict": "CORRECT", "justification": "y",,,}'
    v = parse_judge_response(raw, "vsp")
    assert v.is_correct and v.extracted_answer == "CVSS:3.1/AV:N"


def test_empty_response_skipped():
    v = parse_judge_response("   ", "mcq")
    assert v.skipped and not v.is_correct


def test_error_prefix_skipped():
    v = parse_judge_response("ERROR: timeout", "mcq")
    assert v.skipped


def test_content_filter_skipped():
    v = parse_judge_response("blocked by content management policy", "mcq")
    assert v.skipped


def test_strip_reasoning_removes_think_block():
    assert strip_reasoning("<think>lots of cot</think>\nAnswer: B") == "Answer: B"


def test_strip_reasoning_applies_stop_after_think():
    # Stop applies to the answer portion only, not inside the thinking trace.
    out = strip_reasoning("<think>newlines\nhere</think>B\nextra", stop=["\n"])
    assert out == "B"
