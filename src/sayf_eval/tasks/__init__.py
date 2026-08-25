"""Task registrations. Importing this package populates the registry.

Covers the full 24-task suite: CTI-Bench (HF + TSV), AthenaBench (JSONL),
SECURE, SecEval, CyberMetric, SecBench, MMLU-CS, RedSage, SEvenLLM.
Per-task token budgets and system prompts follow the original pipeline's
standardized config.
"""

from __future__ import annotations

from sayf_eval import datasets as ds
from sayf_eval.registry import register
from sayf_eval.task import Task


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
# Bespoke-loader sources (mirror the identifiers used in datasets.py loaders).
_CYBERMETRIC_HF = "RISys-Lab/Benchmarks_CyberSec_CyberMetrics"
_SECEVAL_HF = "XuanwuAI/SecEval"
_MMLU_HF = "lighteval/mmlu"
_SEVENLLM_HF = "Multilingual-Multimodal-NLP/SEVENLLM-Dataset"
# Original benchmark sources (replace the RISys-Lab mirrors where the data is
# byte-identical). SecBench, SECURE-CWET and RedSage stay on RISys-Lab.
_AI4SEC = "AI4Sec/cti-bench"  # original CTI-Bench (authors' HF dataset)
_CTI_TAA_REPO = "https://github.com/xashru/cti-bench"  # original TAA prompts + gold
_SECURE_ORIG = "https://github.com/aiforsec/SECURE"  # original SECURE (arXiv 2405.20441)
_CYBERMETRIC_ORIG = "https://github.com/cybermetric/CyberMetric"  # original CyberMetric (IEEE CSR 2024)


# ── Declared dataset provenance (see Task.source). One of two neutral shapes
#    (HF dataset or URL); downstream exporters (e.g. the Every Eval Ever
#    converter) map these onto their own source_data schema. ────────────────────
def _hf_source(repo, subset, name, split="test"):
    src = {"type": "hf_dataset", "dataset_name": name, "hf_repo": repo}
    if subset:
        src["subset"] = subset
    if split:
        src["split"] = split
    return src


def _url_source(url, name):
    return {"type": "url", "dataset_name": name, "url": [url]}


def _reg(name, task_type, loader, system_prompt=None, max_tokens=1024, source=None):
    register(
        Task(
            name=name,
            task_type=task_type,
            loader=loader,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            source=source,
        )
    )


# ── CTI-Bench (AI4Sec/cti-bench, original HF) — domain system prompt ──────────
_reg(
    "mcq",
    "mcq",
    ds.make_cti_ai4sec_loader(_AI4SEC, "cti-mcq", "mcq"),
    _CTI,
    1024,
    _hf_source(_AI4SEC, "cti-mcq", "CTI-Bench MCQ"),
)
_reg(
    "rcm",
    "rcm",
    ds.make_cti_ai4sec_loader(_AI4SEC, "cti-rcm", "rcm"),
    _CTI,
    512,
    _hf_source(_AI4SEC, "cti-rcm", "CTI-Bench RCM (CWE mapping)"),
)
_reg(
    "vsp",
    "vsp",
    ds.make_cti_ai4sec_loader(_AI4SEC, "cti-vsp", "vsp"),
    _CTI,
    2048,
    _hf_source(_AI4SEC, "cti-vsp", "CTI-Bench VSP (CVSS)"),
)
_reg(
    "ate",
    "ate",
    ds.make_cti_ai4sec_loader(_AI4SEC, "cti-ate", "ate"),
    _CTI,
    1024,
    _hf_source(_AI4SEC, "cti-ate", "CTI-Bench ATE (ATT&CK)"),
)
_reg("cti_taa", "taa", ds.load_cti_taa, _CTI, 512, _url_source(_CTI_TAA_REPO, "CTI-Bench TAA"))

# ── SECURE (ICS/OT MCQ) — Prompt prebuilt, no system prompt ──────────────────
_reg(
    "secure_maet",
    "secure",
    ds.make_secure_orig_loader("MAET", "secure"),
    None,
    256,
    _url_source(_SECURE_ORIG, "SECURE MAET"),
)
_reg(
    "secure_cwet",
    "secure",
    ds.make_hf_loader("secure_cwet", _SECURE, "CWET", "secure"),
    None,
    256,
    _hf_source(_SECURE, "CWET", "SECURE CWET"),
)
_reg(
    "secure_kcv",
    "secure_kcv",
    ds.make_secure_orig_loader("KCV", "secure_kcv"),
    None,
    256,
    _url_source(_SECURE_ORIG, "SECURE KCV"),
)

# ── SecBench (English MCQ) — RedSage wording, no system prompt ────────────────
_reg(
    "secbench",
    "mcq",
    ds.make_hf_loader("secbench", _SECBENCH, "MCQs_English", "mcq"),
    None,
    256,
    _hf_source(_SECBENCH, "MCQs_English", "SecBench MCQ (English)"),
)

# ── RedSage MCQ (5 subsets) — no system prompt ───────────────────────────────
for _key, _sub in [
    ("redsage_frameworks", "cybersecurity_knowledge_frameworks"),
    ("redsage_generals", "cybersecurity_knowledge_generals"),
    ("redsage_skills", "cybersecurity_skills"),
    ("redsage_cli", "cybersecurity_tools_cli"),
    ("redsage_kali", "cybersecurity_tools_kali"),
]:
    _reg(
        _key,
        "mcq",
        ds.make_hf_loader(_key, _REDSAGE, _sub, "mcq"),
        None,
        256,
        _hf_source(_REDSAGE, _sub, f"RedSage: {_sub}"),
    )

# ── AthenaBench (GitHub JSONL) — no system prompt ────────────────────────────
_reg(
    "ckt",
    "ckt",
    ds.make_athena_loader(f"{_ATHENA}/athena-cti-ckt-3k.jsonl", "ckt"),
    None,
    1024,
    _url_source(f"{_ATHENA}/athena-cti-ckt-3k.jsonl", "AthenaBench CKT"),
)
_reg(
    "rms",
    "rms",
    ds.make_athena_loader(f"{_ATHENA}/athena-cti-rms.jsonl", "rms"),
    None,
    512,
    _url_source(f"{_ATHENA}/athena-cti-rms.jsonl", "AthenaBench RMS"),
)
_reg(
    "taa",
    "taa",
    ds.make_athena_loader(f"{_ATHENA}/athena-cti-taa.jsonl", "taa"),
    None,
    512,
    _url_source(f"{_ATHENA}/athena-cti-taa.jsonl", "AthenaBench TAA"),
)
_reg(
    "athena_ate",
    "ate",
    ds.make_athena_loader(f"{_ATHENA}/athena-cti-ate.jsonl", "ate"),
    None,
    256,
    _url_source(f"{_ATHENA}/athena-cti-ate.jsonl", "AthenaBench ATE"),
)
_reg(
    "athena_rcm",
    "rcm",
    ds.make_athena_loader(f"{_ATHENA}/athena-cti-rcm.jsonl", "rcm"),
    None,
    1024,
    _url_source(f"{_ATHENA}/athena-cti-rcm.jsonl", "AthenaBench RCM"),
)
_reg(
    "athena_vsp",
    "vsp",
    ds.make_athena_loader(f"{_ATHENA}/athena-cti-vsp.jsonl", "vsp"),
    None,
    1024,
    _url_source(f"{_ATHENA}/athena-cti-vsp.jsonl", "AthenaBench VSP"),
)

# ── Other MCQ benchmarks ─────────────────────────────────────────────────────
# SecEval ships a single questions.json at the HF repo root (no subset/split).
_reg("seceval", "seceval", ds.load_seceval, _SECEVAL, 256, _hf_source(_SECEVAL_HF, None, "SecEval", split=None))
_reg(
    "cybermetric",
    "mcq",
    ds.load_cybermetric,
    _CYBERMETRIC,
    256,
    _url_source(_CYBERMETRIC_ORIG, "CyberMetric-500"),
)
_reg(
    "mmlu-cs",
    "mcq",
    ds.load_mmlu_cs,
    None,
    512,
    _hf_source(_MMLU_HF, "computer_security", "MMLU computer_security (5-shot)"),
)

# ── SEvenLLM (open-ended structured CTI extraction / analysis) ────────────────
# Judged semantically; open-ended JSON/text outputs need a larger budget.
# English subset of test.jsonl (non-Chinese rows kept by the loader).
_reg("sevenllm", "sevenllm", ds.load_sevenllm, None, 2000, _hf_source(_SEVENLLM_HF, None, "SEvenLLM-Bench (English)"))
