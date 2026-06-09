"""Authoritative Tier-1 parity: new loaders vs the CURRENT original collectors.

Imports the original ``run_inference_benchmarks`` and stubs out generation
(``batch_generate`` / ``generate_response``) so each collector still builds and
writes its exact ``prompt`` + ``ground_truth`` but makes no model/API call.
Diffs those against the new loaders per ``index``. Deterministic, no API.

This supersedes the stale committed ``outputs_test_5samples`` (which was produced
across older repo revisions); here both sides run current code.

Usage (compute node — needs datasets + HF/GitHub network):
    python tests/parity_vs_current.py [--samples 5] [--tasks mcq rcm ...]
"""

import argparse
import json
import os
import sys
import tempfile


ORIG_DIR = "/export/home/aberriche/BenchBench/BenchmarkingSecBenchmarks/unified-benchmark-pipeline"

# task -> how the original collects it (mirrors run_inference_benchmarks.main()).
_HF = "RISys-Lab/Benchmarks_CyberSec_CTI-Bench"
_SECURE = "RISys-Lab/Benchmarks_CyberSec_SECURE"
_SECBENCH = "RISys-Lab/Benchmarks_CyberSec_SecBench"
_REDSAGE = "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ"
_ATHENA = "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark"

HF_TASKS = {
    "mcq": (_HF, "cti-mcq"),
    "rcm": (_HF, "cti-rcm"),
    "vsp": (_HF, "cti-vsp"),
    "ate": (_HF, "cti-ate"),
    "secure_maet": (_SECURE, "MAET"),
    "secure_cwet": (_SECURE, "CWET"),
    "secure_kcv": (_SECURE, "KCV"),
    "secbench": (_SECBENCH, "MCQs_English"),
    "redsage_frameworks": (_REDSAGE, "cybersecurity_knowledge_frameworks"),
    "redsage_generals": (_REDSAGE, "cybersecurity_knowledge_generals"),
    "redsage_skills": (_REDSAGE, "cybersecurity_skills"),
    "redsage_cli": (_REDSAGE, "cybersecurity_tools_cli"),
    "redsage_kali": (_REDSAGE, "cybersecurity_tools_kali"),
}
JSONL_TASKS = {
    "ckt": f"{_ATHENA}/athena-cti-ckt-3k.jsonl",
    "rms": f"{_ATHENA}/athena-cti-rms.jsonl",
    "taa": f"{_ATHENA}/athena-cti-taa.jsonl",
    "athena_ate": f"{_ATHENA}/athena-cti-ate.jsonl",
    "athena_rcm": f"{_ATHENA}/athena-cti-rcm.jsonl",
    "athena_vsp": f"{_ATHENA}/athena-cti-vsp.jsonl",
}
TSV_TASKS = {"cti_taa": "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-taa.tsv"}
SPECIAL = {"mmlu-cs", "seceval", "cybermetric"}
# original stored seceval prompt as a chat-message LIST; new loader folds it to
# text — compare ground truth only.
PROMPT_DIFF_SKIP = {"seceval"}

ALL_TASKS = list(HF_TASKS) + list(JSONL_TASKS) + list(TSV_TASKS) + sorted(SPECIAL)


def collect_original(orig, task: str, out: str, n: int) -> None:
    """Invoke the current original collector for `task`, generation stubbed."""
    if task in HF_TASKS:
        dn, sub = HF_TASKS[task]
        orig.collect_huggingface_benchmark(task, dn, sub, None, None, out, n, 1024)
    elif task in JSONL_TASKS:
        orig.collect_athenabench_jsonl(task, JSONL_TASKS[task], None, None, out, n, 1024)
    elif task in TSV_TASKS:
        orig.collect_ctibench_tsv(task, TSV_TASKS[task], None, None, out, n, 1024)
    elif task == "mmlu-cs":
        orig.collect_mmlu_cs(None, None, out, n, 1024)
    elif task == "seceval":
        orig.collect_seceval(None, None, out, n, 1024)
    elif task == "cybermetric":
        orig.collect_cybermetric(None, None, out, n, 1024)
    else:
        raise KeyError(task)


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(s) -> str:
    return s.strip() if isinstance(s, str) else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    args = ap.parse_args()

    sys.path.insert(0, ORIG_DIR)
    import run_inference_benchmarks as orig

    # Stub generation: collectors still build/write prompts, but call no model.
    orig.batch_generate = lambda items, *a, **k: ["" for _ in items]
    orig.generate_response = lambda *a, **k: ""

    import sayf_eval.tasks  # noqa: F401
    from sayf_eval.registry import available_tasks, get_task

    rows = []
    overall_ok = True
    for task in args.tasks:
        if task not in available_tasks():
            rows.append((task, "—", "not in new registry", ""))
            continue
        out = os.path.join(tempfile.mkdtemp(), f"{task}.jsonl")
        try:
            collect_original(orig, task, out, args.samples)
        except Exception as e:
            rows.append((task, "ERR", f"original collector failed: {e}", ""))
            overall_ok = False
            continue
        ref = read_jsonl(out)
        if not ref:
            rows.append((task, "—", "original produced 0 rows", ""))
            continue
        try:
            new = {s.index: s for s in get_task(task).load(max_samples=len(ref))}
        except Exception as e:
            rows.append((task, "ERR", f"new loader failed: {e}", ""))
            overall_ok = False
            continue

        gt_ok = p_ok = p_checked = 0
        first = ""
        for r in ref:
            s = new.get(r.get("index"))
            if s is None:
                continue
            if norm(s.target) == norm(str(r.get("ground_truth", ""))):
                gt_ok += 1
            if task not in PROMPT_DIFF_SKIP:
                p_checked += 1
                if norm(s.prompt) == norm(r.get("prompt", "")):
                    p_ok += 1
                elif not first:
                    first = (
                        f"\n    new[:160]={norm(s.prompt)[:160]!r}\n    ref[:160]={norm(r.get('prompt', ''))[:160]!r}"
                    )
        n = len(ref)
        if task in PROMPT_DIFF_SKIP:
            ok = gt_ok == n
            detail = f"prompt SKIP (intentional); gt {gt_ok}/{n}"
        else:
            ok = p_ok == p_checked and gt_ok == n
            detail = f"prompt {p_ok}/{p_checked}; gt {gt_ok}/{n}"
        overall_ok = overall_ok and ok
        rows.append((task, "PASS" if ok else "DIFF", detail, first))

    print(f"{'task':20s} {'verdict':7s} detail")
    for task, verdict, detail, diff in rows:
        print(f"{task:20s} {verdict:7s} {detail}{diff}")
    print("\nPARITY (vs current original collectors):", "PASS" if overall_ok else "DIFFERENCES FOUND")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
