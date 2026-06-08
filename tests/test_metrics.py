"""Metric tests — VSP MAD, set P/R/F1, corpus accuracy + denominator policy."""

from seceval.metrics import (
    calculate_vsp_mad,
    compute_ate_metrics,
    score_corpus,
    set_prf1,
    split_id_set,
    parent_only,
)


def test_vsp_mad_identical_is_zero():
    v = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert calculate_vsp_mad(v, v) == 0.0


def test_vsp_mad_prefix_normalization():
    a = "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    b = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert calculate_vsp_mad(a, b) == 0.0


def test_vsp_mad_invalid_is_max():
    assert calculate_vsp_mad("not-a-vector", "also-bad") == 10.0


def test_set_prf1_exact():
    r = set_prf1({"A", "B"}, {"A", "B"})
    assert r["f1"] == 1.0 and r["exact_match"] == 1


def test_set_prf1_partial():
    r = set_prf1({"A"}, {"A", "B"})
    assert r["precision"] == 1.0 and r["recall"] == 0.5 and r["exact_match"] == 0


def test_parent_only_strips_subtechnique():
    assert parent_only(split_id_set("T1059.001,T1059.003,T1027")) == {"T1059", "T1027"}


def test_ate_micro_f1():
    rows = [
        {"extracted_answer": "T1059.001", "ground_truth": "T1059"},   # parent match
        {"extracted_answer": "T1027", "ground_truth": "T1027,T1055"}, # 1 tp, 1 fn
    ]
    m = compute_ate_metrics(rows, parent=True)
    assert m["tp_total"] == 2 and m["fn_total"] == 1 and m["fp_total"] == 0


def test_corpus_accuracy_denominator_excludes_skipped_only():
    rows = [
        {"is_correct": True, "skipped": False},
        {"is_correct": False, "skipped": False},   # unparseable model answer = incorrect, still counted
        {"is_correct": False, "skipped": True},     # judge failure = excluded
    ]
    r = score_corpus("mcq", rows)
    assert r["correct"] == 1 and r["total"] == 2 and r["skipped"] == 1
    assert r["accuracy"] == 0.5


def test_corpus_vsp_adds_mad():
    rows = [
        {"is_correct": True, "skipped": False,
         "extracted_answer": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
         "ground_truth": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"},
    ]
    r = score_corpus("vsp", rows)
    assert "mad" in r and r["mad"] == 0.0 and r["extraction_success"] == 1
