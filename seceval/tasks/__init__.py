"""Task registrations. Importing this package populates the registry.

Covers the full ~24-task suite: CTI-Bench (HF + TSV), AthenaBench (JSONL),
SECURE, SecEval, CyberMetric, SecBench, MMLU-CS, RedSage, CISSP. Per-task token
budgets and system prompts follow the original pipeline's standardized config.
"""

from __future__ import annotations

from seceval import datasets as ds
from seceval.registry import register
from seceval.task import Task

# System prompts (only where the benchmark specifies one).
_CTI = "You are a cybersecurity expert specializing in cyberthreat intelligence."
_CYBERMETRIC = "You are a security expert who answers questions."
_SECEVAL = (
    "Below are multiple-choice questions concerning cybersecurity. Please select "
    "the correct answers and respond with the letters ABCD only."
)

_HF = "RISys-Lab/Benchmarks_CyberSec_CTI-Bench"
_SECURE = "RISys-Lab/Benchmarks_CyberSec_SECURE"
_SECBENCH = "RISys-Lab/Benchmarks_CyberSec_SecBench"
_REDSAGE = "RISys-Lab/Benchmarks_CyberSec_RedSageMCQ"
_ATHENA = "https://github.com/Athena-Software-Group/athenabench/raw/main/benchmark"


def _reg(name, task_type, loader, system_prompt=None, max_tokens=1024):
    register(Task(
        name=name, task_type=task_type, loader=loader,
        system_prompt=system_prompt, max_tokens=max_tokens,
    ))


# ── CTI-Bench (RISys-Lab HF) — domain system prompt ──────────────────────────
_reg("mcq", "mcq", ds.make_hf_loader("mcq", _HF, "cti-mcq", "mcq"), _CTI, 1024)
_reg("rcm", "rcm", ds.make_hf_loader("rcm", _HF, "cti-rcm", "rcm"), _CTI, 512)
_reg("vsp", "vsp", ds.make_hf_loader("vsp", _HF, "cti-vsp", "vsp"), _CTI, 2048)
_reg("ate", "ate", ds.make_hf_loader("ate", _HF, "cti-ate", "ate"), _CTI, 1024)
_reg("cti_taa", "taa", ds.load_cti_taa, _CTI, 512)

# ── SECURE (ICS/OT MCQ) — Prompt prebuilt, no system prompt ──────────────────
_reg("secure_maet", "secure", ds.make_hf_loader("secure_maet", _SECURE, "MAET", "secure"), None, 256)
_reg("secure_cwet", "secure", ds.make_hf_loader("secure_cwet", _SECURE, "CWET", "secure"), None, 256)
_reg("secure_kcv", "secure", ds.make_hf_loader("secure_kcv", _SECURE, "KCV", "secure"), None, 256)

# ── SecBench (English MCQ) — RedSage wording, no system prompt ────────────────
_reg("secbench", "mcq", ds.make_hf_loader("secbench", _SECBENCH, "MCQs_English", "mcq"), None, 256)

# ── RedSage MCQ (5 subsets) — no system prompt ───────────────────────────────
for _key, _sub in [
    ("redsage_frameworks", "cybersecurity_knowledge_frameworks"),
    ("redsage_generals", "cybersecurity_knowledge_generals"),
    ("redsage_skills", "cybersecurity_skills"),
    ("redsage_cli", "cybersecurity_tools_cli"),
    ("redsage_kali", "cybersecurity_tools_kali"),
]:
    _reg(_key, "mcq", ds.make_hf_loader(_key, _REDSAGE, _sub, "mcq"), None, 256)

# ── AthenaBench (GitHub JSONL) — no system prompt ────────────────────────────
_reg("ckt", "ckt", ds.make_athena_loader(f"{_ATHENA}/athena-cti-ckt-3k.jsonl", "ckt"), None, 1024)
_reg("rms", "rms", ds.make_athena_loader(f"{_ATHENA}/athena-cti-rms.jsonl", "rms"), None, 512)
_reg("taa", "taa", ds.make_athena_loader(f"{_ATHENA}/athena-cti-taa.jsonl", "taa"), None, 512)
_reg("athena_ate", "ate", ds.make_athena_loader(f"{_ATHENA}/athena-cti-ate.jsonl", "ate"), None, 256)
_reg("athena_rcm", "rcm", ds.make_athena_loader(f"{_ATHENA}/athena-cti-rcm.jsonl", "rcm"), None, 1024)
_reg("athena_vsp", "vsp", ds.make_athena_loader(f"{_ATHENA}/athena-cti-vsp.jsonl", "vsp"), None, 1024)

# ── Other MCQ benchmarks ─────────────────────────────────────────────────────
_reg("seceval", "seceval", ds.load_seceval, _SECEVAL, 256)
_reg("cybermetric", "mcq", ds.load_cybermetric, _CYBERMETRIC, 256)
_reg("mmlu-cs", "mcq", ds.load_mmlu_cs, None, 512)
_reg("cissp", "mcq", ds.load_cissp, None, 1024)
