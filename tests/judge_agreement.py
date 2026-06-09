"""Tier-4 judge-agreement parity: original scoring pipeline vs the new scorer.

Feeds identical stored model responses through BOTH:
  - original: run_evaluate_llm_judge.create_judge_prompt + parse_judge_response
  - new:      sayf_eval.judge_prompts.create_judge_prompt + scorer.parse_judge_response
using the SAME Azure gpt-5.4 judge and the SAME transport (so the only variables
are the prompt-construction and parse code, both ported verbatim).

Reports, per task and overall:
  - prompt-identity rate  (orig judge prompt == new judge prompt, byte-for-byte)
  - verdict-agreement rate (orig is_correct == new is_correct)
  - per-task accuracy (orig vs new) and the delta
  - Cohen's kappa on the correct/incorrect labels

When the two prompts are byte-identical the judge is called once and both
parsers read the same text (so disagreement there is pure parse-code, expected 0);
when they differ each prompt is judged separately (end-to-end).

Usage (compute node, needs AZURE_API_KEY + datasets/network for nothing — reads
local response JSONL only):
    AZURE_API_KEY=... python tests/judge_agreement.py \
        --response-dir <dir> [--per-task 15] [--tasks mcq rcm ...]
"""

import argparse
import json
import os
import sys


AZURE_ENDPOINT = "https://qcri-cyber-cx-ai-03-eus2.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"
ORIG_DIR = "/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/unified-benchmark-pipeline"


def orig_choices_block(meta: dict):
    """Replicate run_evaluate_llm_judge's extra_context choices construction."""
    ch = (meta or {}).get("choices")
    if ch is None:
        return None
    if isinstance(ch, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(ch.items()))
    return ch  # list passed through as-is (original behavior)


def strip_think(text: str) -> str:
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def msg(p: str):
    return [{"role": "user", "content": p}]


def kappa(orig: list[bool], new: list[bool]) -> float:
    n = len(orig)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in zip(orig, new) if a == b) / n
    # marginal probs
    oa = sum(orig) / n
    na = sum(new) / n
    pe = oa * na + (1 - oa) * (1 - na)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--response-dir", required=True)
    ap.add_argument("--per-task", type=int, default=15)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    if not os.environ.get("AZURE_API_KEY"):
        sys.exit("AZURE_API_KEY not set")

    sys.path.insert(0, ORIG_DIR)
    import run_evaluate_llm_judge as O

    from sayf_eval.judge_prompts import create_judge_prompt as new_jp
    from sayf_eval.model import GenParams, Model
    from sayf_eval.scorer import format_choices, strip_reasoning
    from sayf_eval.scorer import parse_judge_response as new_parse

    judge = Model(
        model=f"azure/{AZURE_DEPLOYMENT}",
        api_key=os.environ["AZURE_API_KEY"],
        api_version=AZURE_API_VERSION,
        extra={"api_base": AZURE_ENDPOINT},
        concurrency=args.concurrency,
    )
    params = GenParams(max_tokens=args.max_tokens)

    files = {
        fn[: -len("_responses.jsonl")]: os.path.join(args.response_dir, fn)
        for fn in os.listdir(args.response_dir)
        if fn.endswith("_responses.jsonl")
    }
    tasks = args.tasks or sorted(files)

    # ---- build all (orig_prompt, new_prompt) pairs across tasks ----
    items = []  # (task, task_type, orig_prompt, new_prompt)
    for task in tasks:
        if task not in files:
            continue
        with open(files[task]) as f:
            rows = [json.loads(line) for line in f if line.strip()][: args.per_task]
        for r in rows:
            meta = r.get("metadata", {}) or {}
            tt = meta.get("task_type") or O.get_task_type(task)
            q = r.get("prompt", "")
            if isinstance(q, list):  # original stored chat-list (seceval): take last user turn
                q = next((m.get("content", "") for m in reversed(q) if m.get("role") == "user"), "")
            gt = str(r.get("ground_truth", ""))
            ans = strip_think(r.get("model_response", ""))

            ob = orig_choices_block(meta)
            oc = {"choices": ob} if ob is not None else None
            op = O.create_judge_prompt(tt, q, ans, gt, oc)

            raw_ch = meta.get("choices")
            nc = {"choices": format_choices(raw_ch)} if raw_ch is not None else {}
            np_ = new_jp(tt, q, strip_reasoning(ans), gt, nc)

            items.append((task, tt, op, np_))

    n = len(items)
    print(f">> {n} samples across {len({i[0] for i in items})} tasks; judging via {AZURE_DEPLOYMENT}")

    # ---- judge: one call per orig prompt; extra calls only where prompts differ ----
    orig_texts = [r.text for r in judge.generate_batch([msg(it[2]) for it in items], params)]
    diff_idx = [i for i, it in enumerate(items) if it[2] != it[3]]
    new_texts = list(orig_texts)
    if diff_idx:
        dt = [r.text for r in judge.generate_batch([msg(items[i][3]) for i in diff_idx], params)]
        for j, i in enumerate(diff_idx):
            new_texts[i] = dt[j]

    # ---- compare ----
    per_task: dict[str, dict] = {}
    ko, kn = [], []  # kappa labels over comparable (non-skipped both) samples
    for (task, tt, op, npr), ot, nt in zip(items, orig_texts, new_texts):
        d = per_task.setdefault(
            task,
            {"n": 0, "pmatch": 0, "agree": 0, "cmp": 0, "o_correct": 0, "o_total": 0, "n_correct": 0, "n_total": 0},
        )
        d["n"] += 1
        if op == npr:
            d["pmatch"] += 1
        ov = O.parse_judge_response(ot, tt)
        nv = new_parse(nt, tt)
        o_skip = bool(ov.get("skipped"))
        if not o_skip:
            d["o_total"] += 1
            d["o_correct"] += int(ov.get("is_correct", False))
        if not nv.skipped:
            d["n_total"] += 1
            d["n_correct"] += int(nv.is_correct)
        if not o_skip and not nv.skipped:
            d["cmp"] += 1  # comparable: neither pipeline skipped
            d["agree"] += int(bool(ov.get("is_correct")) == nv.is_correct)
            ko.append(bool(ov.get("is_correct")))
            kn.append(nv.is_correct)

    # ---- report ----
    print(f"\n{'task':18s} {'n':>3s} {'pmatch':>8s} {'agree':>8s} {'acc_orig':>9s} {'acc_new':>8s}")
    tot = {"n": 0, "pmatch": 0, "agree": 0, "cmp": 0}
    for task in sorted(per_task):
        d = per_task[task]
        acc_o = d["o_correct"] / d["o_total"] if d["o_total"] else 0.0
        acc_n = d["n_correct"] / d["n_total"] if d["n_total"] else 0.0
        print(
            f"{task:18s} {d['n']:>3d} {d['pmatch']:>5d}/{d['n']:<2d} "
            f"{d['agree']:>5d}/{d['cmp']:<2d} {acc_o:>9.3f} {acc_n:>8.3f}"
        )
        for k in tot:
            tot[k] += d[k]

    pid = tot["pmatch"] / tot["n"] if tot["n"] else 0.0
    agr = tot["agree"] / tot["cmp"] if tot["cmp"] else 0.0
    print(
        f"\nOVERALL  n={tot['n']}  prompt-identity={pid:.1%} ({tot['pmatch']}/{tot['n']})  "
        f"verdict-agreement={agr:.1%} ({tot['agree']}/{tot['cmp']})  "
        f"kappa={kappa(ko, kn):.3f}"
    )
    print("(agreement base = samples neither pipeline marked judge-skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
