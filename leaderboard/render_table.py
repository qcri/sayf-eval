"""Render the leaderboard markdown from committed records (no private data needed).

Reads leaderboard/leaderboard.json + the per-model results_*.json and prints:
  1. a ranked summary (model, judge, average strict-verdict accuracy)
  2. the full per-task x per-model table, grouped by benchmark family, using each
     task's headline metric (accuracy, or VSP MAD / AthenaBench MAD-norm / RMS F1).

Usage: python leaderboard/render_table.py <leaderboard_dir>
"""

import json
import sys
from pathlib import Path


# benchmark family -> [(task_key, display, metric_kind)]
GROUPS = [
    (
        "CTI-Bench",
        [
            ("mcq", "MCQ", "acc"),
            ("rcm", "RCM", "acc"),
            ("vsp", "VSP", "mad"),
            ("ate", "ATE", "acc"),
            ("cti_taa", "TAA", "acc"),
        ],
    ),
    (
        "AthenaBench",
        [
            ("ckt", "CKT", "acc"),
            ("rms", "RMS", "f1"),
            ("taa", "TAA", "acc"),
            ("athena_ate", "ATE", "acc"),
            ("athena_rcm", "RCM", "acc"),
            ("athena_vsp", "VSP", "mad_norm"),
        ],
    ),
    ("SECURE", [("secure_maet", "MAET", "acc"), ("secure_cwet", "CWET", "acc"), ("secure_kcv", "KCV", "acc")]),
    (
        "RedSage-MCQ",
        [
            ("redsage_frameworks", "FW", "acc"),
            ("redsage_generals", "GEN", "acc"),
            ("redsage_skills", "Skills", "acc"),
            ("redsage_cli", "CLI", "acc"),
            ("redsage_kali", "Kali", "acc"),
        ],
    ),
    ("CyberMetric", [("cybermetric", "CyberMetric", "acc")]),
    ("MMLU-CS", [("mmlu-cs", "MMLU-CS", "acc")]),
    ("SecBench", [("secbench", "SecBench", "acc")]),
    ("SecEval", [("seceval", "SecEval", "acc")]),
    ("SEvenLLM", [("sevenllm", "SEvenLLM", "acc")]),
]
METRIC_SUP = {"acc": "Acc", "mad": "MAD↓", "mad_norm": "MAD-norm", "f1": "F1"}
LOWER_BETTER = {"mad"}

# compact column label per real id (id -> short header); legend keeps the full id.
SHORT = {
    "claude-sonnet-4-6": "Sonnet-4.6",
    "gpt-5.4": "GPT-5.4",
    "gemma-4-31B-it": "Gemma-4-31B",
    "Qwen/Qwen3.6-35B-A3B": "Qwen3.6-35B",
    "Llama-Primus-Nemotron-70B-Instruct": "Primus-Nemo-70B",
    "RISys-Lab/RedSage-Qwen3-8B-DPO": "RedSage-8B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "openai/gpt-oss-20b": "GPT-oss-20B",
    "fdtn-ai/Foundation-Sec-8B-Instruct": "Found-Sec-8B",
    "trendmicro-ailab/Llama-Primus-Merged": "Primus-Merged",
}


def val(rec, task, kind):
    t = rec["results"].get(task)
    if t is None:
        return None
    if kind == "mad":
        return t.get("mad")
    # acc / mad_norm / f1: read then scale, None-safe if a record omits the key
    v = t.get({"acc": "accuracy"}.get(kind, kind))
    return v * 100 if v is not None else None


def fmt(v, kind, bold=False):
    if v is None:
        return "--"
    s = f"{v:.2f}" if kind == "mad" else f"{v:.1f}"
    return f"**{s}**" if bold else s


def main(lb=None):
    # Resolve the leaderboard dir at call time (not import) so importing this
    # module and calling main() programmatically doesn't pick up unrelated argv.
    if lb is None:
        lb = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("leaderboard")
    lb = Path(lb)
    # Records are written UTF-8 (see save_record); read/emit UTF-8 explicitly so
    # the non-ASCII glyphs (≡, —, ↓, ×, ·) survive a non-UTF-8 locale (LANG=C).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    idx = json.loads((lb / "leaderboard.json").read_text(encoding="utf-8"))
    order = [m["model"] for m in idx["models"]]  # ranked by avg desc
    recs = {}
    for m in idx["models"]:
        recs[m["model"]] = json.loads((lb / m["record"]).read_text(encoding="utf-8"))
    avg = {m["model"]: m["avg_accuracy_pct"] for m in idx["models"]}
    # Derive the headline counts from the data so the caption can't drift.
    n_models = len(order)
    n_tasks = len(recs[order[0]]["results"]) if order else 0

    out = []
    # 1) ranked summary
    out.append("### Ranked summary")
    out.append("")
    out.append("| Rank | Model | Avg accuracy (%) |")
    out.append("|---:|---|---:|")
    for i, mid in enumerate(order, 1):
        out.append(f"| {i} | `{mid}` | {avg[mid]:.1f} |")
    out.append("")
    out.append(
        f"_Judge: `{idx['judge']}` · single extract-and-verdict run · "
        f"{n_models} models × {n_tasks} sub-tasks · compiled {idx['created_at'][:10]}._"
    )
    out.append("")

    # 2) full per-task table
    heads = [SHORT.get(m, m) for m in order]
    out.append("### Full results")
    out.append("")
    out.append("| Task | " + " | ".join(heads) + " |")
    out.append("|" + "---|" * (len(heads) + 1))
    for family, tasks in GROUPS:
        out.append(f"| _{family}_ | " + " | ".join([""] * len(heads)) + " |")
        for task, disp, kind in tasks:
            vals = {m: val(recs[m], task, kind) for m in order}
            finite = [v for v in vals.values() if v is not None]
            best = (min(finite) if kind in LOWER_BETTER else max(finite)) if finite else None
            cells = []
            for m in order:
                v = vals[m]
                cells.append(fmt(v, kind, bold=(v is not None and best is not None and abs(v - best) < 1e-9)))
            out.append(f"| {disp} <sup>{METRIC_SUP[kind]}</sup> | " + " | ".join(cells) + " |")
    # average row
    cells = []
    best_avg = max(avg.values()) if avg else None
    for m in order:
        cells.append(fmt(avg[m], "acc", bold=(best_avg is not None and abs(avg[m] - best_avg) < 1e-9)))
    out.append("| **Average** <sup>Acc</sup> | " + " | ".join(cells) + " |")
    out.append("")
    # legend
    out.append(
        "<sup>Acc</sup> strict-verdict accuracy (%) · "
        "<sup>MAD↓</sup> CVSS mean-abs-deviation (lower better) · "
        "<sup>MAD-norm</sup> max(0,1−MAD/7.7)×100 · "
        "<sup>F1</sup> macro-F1 over extracted IDs (%). **Bold** = best per task."
    )
    out.append("")
    out.append("**Column legend (short → model id):** " + " · ".join(f"{SHORT.get(m, m)} = `{m}`" for m in order))
    print("\n".join(out))


if __name__ == "__main__":
    main()
