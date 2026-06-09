"""Tier-1 parity: do the new loaders reproduce the original collectors' prompts?

Compares each new-loader Sample against the original harness's committed outputs
(`BenchmarkingSecBenchmarks/outputs_test_5samples/<task>_responses.jsonl`),
matching on `index` and diffing `prompt` + `ground_truth`. Deterministic, no API.

Usage:
    python tests/parity_prompts.py [--ref <dir>] [--tasks mcq rcm ...]
"""

import argparse
import json
import os
import sys

import sayf_eval.tasks  # noqa: F401 — registers tasks
from sayf_eval.registry import available_tasks, get_task


DEFAULT_REF = "/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/outputs_test_5samples"

# Reference-file stem -> registry task name (here they coincide).
# Intentional, documented divergences are skipped for the prompt diff:
#   - seceval: original stored a chat-message LIST as `prompt`; the new loader
#     folds the 1-shot demo into prompt TEXT (equivalent, not byte-identical).
#   - cissp: needs SECEVAL_CISSP_PATH; no public reference.
PROMPT_DIFF_SKIP = {"seceval", "cissp"}


def load_ref(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s) -> str:
    # prompt may be a list (original chat-style); render to compare GT only.
    if not isinstance(s, str):
        return ""
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DEFAULT_REF)
    ap.add_argument("--tasks", nargs="+", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.ref):
        sys.exit(f"reference dir not found: {args.ref}")

    ref_files = {
        fn[: -len("_responses.jsonl")]: os.path.join(args.ref, fn)
        for fn in os.listdir(args.ref)
        if fn.endswith("_responses.jsonl")
    }
    tasks = args.tasks or sorted(ref_files)

    overall_ok = True
    rows = []
    for task in tasks:
        if task not in ref_files:
            rows.append((task, "—", "no reference file", ""))
            continue
        if task not in available_tasks():
            rows.append((task, "—", "not in registry", ""))
            continue

        ref = load_ref(ref_files[task])
        n = len(ref)
        try:
            samples = get_task(task).load(max_samples=n)
        except Exception as e:  # loader needs a path / network hiccup
            rows.append((task, f"0/{n}", f"loader error: {e}", ""))
            overall_ok = False
            continue

        by_idx = {s.index: s for s in samples}
        gt_match = 0
        prompt_match = 0
        prompt_checked = 0
        first_diff = ""
        for r in ref:
            idx = r.get("index")
            s = by_idx.get(idx)
            if s is None:
                continue
            if norm(s.target) == norm(str(r.get("ground_truth", ""))):
                gt_match += 1
            if task not in PROMPT_DIFF_SKIP:
                prompt_checked += 1
                if norm(s.prompt) == norm(r.get("prompt", "")):
                    prompt_match += 1
                elif not first_diff:
                    first_diff = (
                        f"\n    new[:160]={norm(s.prompt)[:160]!r}\n    ref[:160]={norm(r.get('prompt', ''))[:160]!r}"
                    )

        gt_tag = f"gt {gt_match}/{n}"
        if task in PROMPT_DIFF_SKIP:
            p_tag = "prompt SKIP (intentional divergence)"
            ok = gt_match == n
        else:
            p_tag = f"prompt {prompt_match}/{prompt_checked}"
            ok = (prompt_match == prompt_checked) and (gt_match == n)
        overall_ok = overall_ok and ok
        rows.append((task, "PASS" if ok else "DIFF", f"{p_tag}; {gt_tag}", first_diff))

    print(f"{'task':18s} {'verdict':6s} detail")
    for task, verdict, detail, diff in rows:
        print(f"{task:18s} {verdict:6s} {detail}{diff}")
    print("\nPARITY:", "PASS" if overall_ok else "DIFFERENCES FOUND (review above)")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
