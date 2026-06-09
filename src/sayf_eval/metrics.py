"""Metrics — corpus-level aggregation (lighteval-shaped two-level split).

Sample level is the judge verdict (see ``scorer.py``). This module aggregates a
list of per-sample rows into the task score, mirroring
``run_evaluate_llm_judge.py`` lines 762-806.

Invariants:
- **Denominator policy:** accuracy = correct / total over all *attempted* items.
  Unparseable/empty model answers count as incorrect; only judge-failure
  ``skipped`` rows are removed from both numerator and denominator.
- **Direction:** accuracy is higher-is-better; VSP ``mad`` is lower-is-better.

Threat-actor (``taa``) alias resolution is delegated to the judge in natural
language — there is no hardcoded alias table here (matching the original).
"""

from __future__ import annotations


# A "row" is a dict with at least: is_correct, skipped, extracted_answer, and
# (for structured metrics) ground_truth. Produced by the pipeline from a
# SampleVerdict joined with its Sample target.

_ATE_TYPES = ("ate", "athena_ate")
_VSP_TYPES = ("vsp", "athena_vsp")


# -- ID-set helpers (ported verbatim) ---------------------------------------


def split_id_set(s: str) -> set[str]:
    """Comma-separated canonical ID list → normalized uppercase set.

    Handles ``"NONE"``, empty, and surrounding whitespace.
    """
    if not s or str(s).strip().upper() in ("NONE", ""):
        return set()
    return {part.strip().upper() for part in str(s).split(",") if part.strip()}


def parent_only(id_set: set[str]) -> set[str]:
    """Strip MITRE subtechnique suffix (``T1059.001`` → ``T1059``)."""
    return {tid.split(".")[0] for tid in id_set}


# -- VSP / CVSS MAD (ported verbatim) ---------------------------------------


def calculate_vsp_mad(pred_vector: str, gold_vector: str) -> float:
    """Absolute difference of CVSS v3.1 base scores. 10.0 on parse failure."""
    try:
        from cvss import CVSS3

        def normalize(v: str) -> str:
            v = v.strip()
            if v.startswith("CVSS:3.0/"):
                return v.replace("CVSS:3.0/", "CVSS:3.1/")
            if v.startswith("CVSS:3.1/"):
                return v
            return "CVSS:3.1/" + v

        pred_score = CVSS3(normalize(pred_vector)).scores()[0]
        gold_score = CVSS3(normalize(gold_vector)).scores()[0]
        return round(abs(pred_score - gold_score), 2)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Failed to calculate CVSS MAD: {e}")
        return 10.0


# -- set precision/recall/F1 -------------------------------------------------


def set_prf1(pred_set: set[str], gold_set: set[str]) -> dict:
    """Precision / recall / F1 / exact-match for one prediction-vs-gold pair."""
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "exact_match": int(pred_set == gold_set),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# -- corpus aggregations -----------------------------------------------------


def compute_ate_metrics(rows: list[dict], parent: bool = True) -> dict:
    """Micro-averaged precision/recall/F1 + exact-match for ATE-style tasks."""
    tp_total = fp_total = fn_total = exact = 0
    for r in rows:
        if r.get("skipped"):
            continue
        pred = split_id_set(r.get("extracted_answer", ""))
        gold = split_id_set(r.get("ground_truth", ""))
        if parent:
            pred, gold = parent_only(pred), parent_only(gold)
        tp_total += len(pred & gold)
        fp_total += len(pred - gold)
        fn_total += len(gold - pred)
        if pred == gold:
            exact += 1
    p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "exact_matches": exact,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }


def compute_vsp_metrics(rows: list[dict]) -> dict:
    """Mean MAD between extracted CVSS vectors and ground truth."""
    mads: list[float] = []
    extraction_success = 0
    for r in rows:
        if r.get("skipped"):
            continue
        ext = r.get("extracted_answer", "") or ""
        if ext and ext.upper() != "NONE":
            mads.append(calculate_vsp_mad(ext, r.get("ground_truth", "")))
            extraction_success += 1
        else:
            mads.append(10.0)
    mean_mad = sum(mads) / len(mads) if mads else 10.0
    return {"mad": round(mean_mad, 3), "extraction_success": extraction_success}


def score_corpus(task_type: str, rows: list[dict]) -> dict:
    """Aggregate per-sample rows into the task score.

    accuracy = correct / total over attempted (non-skipped) items, plus the
    task-specific traditional metric (ATE micro-F1, VSP mean MAD).
    """
    correct = sum(1 for r in rows if not r.get("skipped") and r.get("is_correct"))
    total = sum(1 for r in rows if not r.get("skipped"))
    skipped = sum(1 for r in rows if r.get("skipped"))
    results = {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "skipped": skipped,
    }
    tlow = task_type.lower()
    if tlow in _ATE_TYPES:
        results.update(compute_ate_metrics(rows, parent=True))
    elif tlow in _VSP_TYPES:
        results.update(compute_vsp_metrics(rows))
    return results
