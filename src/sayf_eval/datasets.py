"""Dataset loaders — produce ``list[Sample]`` per benchmark.

Each loader normalizes a benchmark's raw rows into :class:`~sayf_eval.task.Sample`
(the full rendered prompt text + ground truth + choices). Prompt construction is
ported from the original ``collect_*`` functions so task semantics are unchanged.
Loaders are lazy (called only when a task runs), so importing the registry does
not trigger downloads.

Most tasks come from two factories — :func:`make_hf_loader` (RISys-Lab HF mirror:
CTI MCQ/RCM/VSP/ATE, SECURE, SecBench, RedSage) and :func:`make_athena_loader`
(AthenaBench GitHub JSONL) — plus dedicated loaders for the few benchmarks with
bespoke prompt formats (SecEval, MMLU-CS 5-shot, CyberMetric, CISSP, CTI-TAA TSV).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.request
from collections.abc import Callable

from sayf_eval.task import Sample


# MCQ instruction wording, ported from collect_huggingface_benchmark.
_MCQ_INSTRUCTION = (
    "You are given multiple choice questions. Answer with the option letter "
    "(A, B, C, D) from the given choices directly."
)
_MCQ_INSTRUCTION_SECBENCH = (
    "You are given multiple choice questions. Answer with the option letter from the given choices directly."
)
_LETTERS = ["A", "B", "C", "D", "E"]


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        return r.read()


def _choices_to_list(choices) -> list[str] | None:
    """Normalize a choices field (dict or list) to letter-prefixed lines."""
    if not choices:
        return None
    if isinstance(choices, dict):
        out = []
        for key in _LETTERS:
            if key in choices and str(choices[key]).strip():
                out.append(f"{key}. {choices[key]}")
        # fall back to whatever keys exist if not A–E
        if not out:
            out = [f"{k}. {v}" for k, v in sorted(choices.items())]
        return out or None
    if isinstance(choices, list):
        return [f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(choices[:5])]
    return None


def _normalize_gt(row: dict) -> str:
    """Ground-truth normalization: GT/solution, else answer/label (int→letter)."""
    gt = row.get("GT") or row.get("solution")
    if gt:
        return str(gt).strip()
    val = row.get("answer")
    if val is None:
        val = row.get("label")
    if val is None:
        return ""
    if isinstance(val, int):
        return _LETTERS[val] if 0 <= val < len(_LETTERS) else str(val)
    return str(val).strip()


def _render_mcq(instruction: str, question: str, choices_list: list[str]) -> str:
    body = instruction + "\n\nQuestion: " + question + "\n"
    body += "\n".join(choices_list) + "\n"
    body += "Answer:"
    return body


# -- factory: RISys-Lab HuggingFace benchmarks ------------------------------


def make_hf_loader(name: str, dataset_name: str, subset: str, task_type: str) -> Callable[[], list[Sample]]:
    """Loader for RISys-Lab HF benchmarks (CTI MCQ/RCM/VSP/ATE, SECURE, SecBench,
    RedSage). MCQ-family rows with choices get instruction + scaffolding; SECURE
    and the structured CTI tasks (RCM/VSP/ATE) use the prebuilt ``Prompt`` as-is.
    """
    is_secure = name.startswith("secure_")
    instruction = _MCQ_INSTRUCTION_SECBENCH if name == "secbench" else _MCQ_INSTRUCTION

    def loader() -> list[Sample]:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, subset, split="test")
        samples: list[Sample] = []
        for idx, row in enumerate(ds):
            question = row.get("Prompt") or row.get("prompt") or row.get("question")
            if not question:
                continue
            gt = _normalize_gt(row)
            choices_list = _choices_to_list(row.get("answers") or row.get("choices") or row.get("options"))
            # Build the MCQ prompt only when choices exist and the benchmark
            # didn't already ship a fully-formatted Prompt (SECURE / structured).
            if choices_list and not is_secure:
                prompt = _render_mcq(instruction, question, choices_list)
            else:
                prompt = question
            samples.append(
                Sample(
                    index=idx,
                    prompt=prompt,
                    target=gt,
                    choices=choices_list,
                    metadata={"task_type": task_type, "subset": subset},
                )
            )
        return samples

    return loader


# -- factory: AthenaBench GitHub JSONL --------------------------------------


def make_athena_loader(url: str, task_type: str) -> Callable[[], list[Sample]]:
    """Loader for AthenaBench JSONL tasks (CKT, RMS, TAA, ATE, RCM, VSP).

    The ``prompt`` field is already fully formatted; CKT carries ``option_a..e``.
    """

    def loader() -> list[Sample]:
        text = _http_get(url).decode("utf-8")
        samples: list[Sample] = []
        idx = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question", "")
            prompt = row.get("prompt", question)
            gt = str(row.get("answer", row.get("correct_answer", ""))).strip()
            choices = None
            if "option_a" in row:
                choices = _choices_to_list({L: row.get(f"option_{L.lower()}", "") for L in _LETTERS})
            samples.append(
                Sample(
                    index=idx,
                    prompt=prompt,
                    target=gt,
                    choices=choices,
                    metadata={"task_type": task_type, "source": url},
                )
            )
            idx += 1
        return samples

    return loader


# -- SecEval (XuanwuAI JSON) -------------------------------------------------

# SecEval ships a system instruction + a 1-shot demonstration. We keep the
# instruction as the task system prompt (registry) and fold the 1-shot example
# into the prompt text so the framework's plain (system, user) message shape is
# preserved without a bespoke message builder.
_SECEVAL_FEWSHOT = (
    "Example —\n"
    "Question: Which mitigation prevent stack overflow bug? "
    "A: Stack Canary. B: ALSR. C: CFI. D: Code Signing.\n"
    "Answer: ABC\n\n"
)


def load_seceval() -> list[Sample]:
    url = "https://huggingface.co/datasets/XuanwuAI/SecEval/resolve/main/questions.json"
    questions = json.loads(_http_get(url).decode("utf-8"))
    samples: list[Sample] = []
    for idx, q in enumerate(questions):
        question = q.get("question", "")
        choices = q.get("choices", [])
        if not question or not choices:
            continue
        body = ("Question: " + question + " " + " ".join(choices)).replace("\n", " ")
        prompt = _SECEVAL_FEWSHOT + body
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target=str(q.get("answer", "")).strip(),
                choices=list(choices),
                metadata={"task_type": "seceval", "id": q.get("id", "")},
            )
        )
    return samples


# -- CTI-Bench TAA (maveryn TSV; no published GT) ---------------------------


def load_cti_taa() -> list[Sample]:
    url = "https://raw.githubusercontent.com/maveryn/cti-bench/main/data/cti-taa.tsv"
    text = _http_get(url).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    cols = reader.fieldnames or []
    gt_col = "GT" if "GT" in cols else ("Solution" if "Solution" in cols else None)
    samples: list[Sample] = []
    for idx, row in enumerate(reader):
        prompt = (row.get("Prompt") or "").strip()
        if not prompt:
            continue
        target = str(row.get(gt_col, "")).strip() if gt_col else ""
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target=target,
                metadata={"task_type": "taa", "subset": "cti-taa"},
            )
        )
    return samples


# -- MMLU computer_security (official 5-shot) -------------------------------


def load_mmlu_cs() -> list[Sample]:
    from datasets import load_dataset

    dev = load_dataset("lighteval/mmlu", "computer_security", split="dev")
    test = load_dataset("lighteval/mmlu", "computer_security", split="test")
    header = "The following are multiple choice questions (with answers) about computer security.\n\n"

    def fmt(sample, include_answer: bool) -> str:
        choices = sample.get("choices", [])
        ans = sample.get("answer")
        text = sample.get("question", "").strip() + "\n"
        for i, opt in enumerate(choices[:4]):
            text += f"{_LETTERS[i]}. {opt}\n"
        text += "Answer:"
        if include_answer and ans is not None:
            text += f" {_LETTERS[ans] if isinstance(ans, int) else ans}"
        return text

    prefix = header + "".join(fmt(s, True) + "\n\n" for s in dev)

    samples: list[Sample] = []
    for idx, s in enumerate(test):
        choices = s.get("choices", [])
        ans = s.get("answer")
        if not choices or ans is None:
            continue
        gt = _LETTERS[ans] if isinstance(ans, int) else str(ans).strip()
        samples.append(
            Sample(
                index=idx,
                prompt=prefix + fmt(s, False),
                target=gt,
                choices=_choices_to_list(list(choices)),
                metadata={"task_type": "mcq", "subset": "computer_security"},
            )
        )
    return samples


# -- CyberMetric-500 ---------------------------------------------------------


def load_cybermetric() -> list[Sample]:
    from datasets import load_dataset

    ds = load_dataset("RISys-Lab/Benchmarks_CyberSec_CyberMetrics", "cyberMetric_500", split="test")
    samples: list[Sample] = []
    for idx, row in enumerate(ds):
        question = row.get("question", "")
        answers = row.get("answers", {}) or row.get("choices", {}) or row.get("options", {}) or {}
        gt = str(row.get("solution", "")).strip()
        if not question:
            continue
        if isinstance(answers, list):
            answers = {_LETTERS[i]: str(o) for i, o in enumerate(answers[:4])}
        if not isinstance(answers, dict) or not answers:
            continue
        options = ", ".join(f"{k}) {v}" for k, v in answers.items())
        prompt = (
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            f"Choose the correct answer (A, B, C, or D) only. "
            f"Always return in this format: 'ANSWER: X'"
        )
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target=gt,
                choices=_choices_to_list(answers),
                metadata={"task_type": "mcq", "subset": "cyberMetric_500"},
            )
        )
    return samples


# -- CISSP (local/remote JSON; path via SAYF_EVAL_CISSP_PATH) ------------------


def load_cissp() -> list[Sample]:
    path = os.environ.get("SAYF_EVAL_CISSP_PATH")
    if not path:
        raise ValueError(
            "CISSP dataset path required — set SAYF_EVAL_CISSP_PATH to a local JSON "
            "file or URL (the CISSP set is not a public HF dataset)."
        )
    if path.startswith("http"):
        data = json.loads(_http_get(path).decode("utf-8"))
    else:
        with open(path) as f:
            data = json.load(f)
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        questions = data.get("questions") or data.get("items") or data.get("data") or []
    else:
        questions = []

    samples: list[Sample] = []
    for idx, q in enumerate(questions):
        question = q.get("question") or q.get("Prompt") or ""
        if not question:
            continue
        choices: dict = {}
        if isinstance(q.get("answers"), dict):
            choices = q["answers"]
        elif isinstance(q.get("options"), list):
            choices = {chr(65 + i): c for i, c in enumerate(q["options"][:4])}
        elif isinstance(q.get("options"), dict):
            choices = q["options"]
        elif isinstance(q.get("choices"), list):
            choices = {chr(65 + i): c for i, c in enumerate(q["choices"][:4])}
        else:
            choices = {L: q[L] for L in ["A", "B", "C", "D"] if L in q}
        gt = ""
        for key in ["solution", "answer", "GT", "correct_answer"]:
            if key in q:
                gt = str(q[key]).strip()
                break
        if not choices or not gt:
            continue
        prompt = f"{question}\n\n"
        for label in sorted(choices.keys()):
            prompt += f"{label}. {choices[label]}\n"
        prompt += "\nAnswer with the letter only:"
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target=gt,
                choices=_choices_to_list(choices),
                metadata={"task_type": "mcq", "domain": q.get("domain", "")},
            )
        )
    return samples


# -- SEvenLLM-Bench (English subset, open-ended structured CTI extraction) ----

# Stanford-Alpaca-style instruction/input wrapper SEvenLLM was fine-tuned on.
# Only the plain variant is used: model-specific chat tokens are added by the
# serving backend's chat template, not baked into the prompt text.
_SEVENLLM_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
# CJK Unified Ideographs — used to keep the English half of the bilingual set.
_CHINESE_CHAR_RE = re.compile(r"[一-鿿]")


def _is_non_chinese(text: str) -> bool:
    return not bool(_CHINESE_CHAR_RE.search(text or ""))


def load_sevenllm() -> list[Sample]:
    """SEvenLLM-Bench English subset (open-ended CTI extraction / analysis).

    Bilingual benchmark (650 EN + 650 ZH); we keep the non-Chinese inputs so the
    judge runs a single English prompt. The HF auto-loader trips on this dataset
    (mixed output types per row), so download ``test.jsonl`` directly.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Multilingual-Multimodal-NLP/SEVENLLM-Dataset",
        filename="test.jsonl",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if _is_non_chinese(r.get("input", ""))]

    samples: list[Sample] = []
    for idx, r in enumerate(rows):
        instruction = r.get("instruction", "")
        input_text = r.get("input", "")
        gt = r.get("output", "")
        target = json.dumps(gt, ensure_ascii=False) if isinstance(gt, (dict, list)) else str(gt)
        samples.append(
            Sample(
                index=idx,
                prompt=_SEVENLLM_TEMPLATE.format(instruction=instruction, input=input_text),
                target=target,
                metadata={"task_type": "sevenllm", "category": r.get("category", "unknown")},
            )
        )
    return samples


# -- Native-Arabic MCQ (local JSONL) -----------------------------------------

# Arabic system prompt + letter-only rendering, verbatim from the project's
# eval_tri_mcq.py (SYS_AR / build_prompt_single). Used for benchmarks that are
# *natively* Arabic (no translation), e.g. arabic-native-cyber-mcq, and for
# Arabic-translated local sets (CISSP-Arabic). The English baseline (--translator
# none) runs them as-is; --translator llm --translator-lang en gives translate-test.
SYS_AR = (
    "أنت خبير في الأمن السيبراني.\n"
    "سيتم تزويدك بسؤال اختيار من متعدد مع أربع خيارات A–D.\n"
    "اختر أفضل إجابة واحدة فقط، وأجب بحرف واحد فقط من (A أو B أو C أو D) بدون أي شرح."
)


def _render_mcq_letter(question: str, choices: dict) -> str:
    """Question + options rendered as one block (ported from build_prompt_single)."""
    text = "Question:\n" + (question or "").strip() + "\n\n"
    for k in ["A", "B", "C", "D"]:
        if k in choices and str(choices[k]).strip():
            text += f"{k}: {choices[k]}\n"
    text += "\nReply ONLY with the letter (A, B, C, or D)."
    return text


def _choices_dict(row: dict) -> dict:
    """Coerce a row's options (dict or list, several key spellings) to {A..D: str}."""
    opts = row.get("options") or row.get("choices") or row.get("answers") or {}
    if isinstance(opts, list):
        return {_LETTERS[i]: str(o) for i, o in enumerate(opts[:5])}
    if isinstance(opts, dict):
        return {str(k).strip().upper(): str(v) for k, v in opts.items()}
    return {}


# -- CyberSecEval (Meta PurpleLlama) LLM-judge subsets ------------------------

# Three CyberSecEval subsets whose canonical scoring is an *LLM judge* (not a
# static analyzer), so they fit this harness's judge directly:
#   * prompt-injection — did the model follow an injected instruction?
#   * mitre            — does the response give offensive ATT&CK uplift?
#   * interpreter      — does the response produce code-interpreter abuse?
# These are *safety* judgments: the "good" outcome is the model staying safe
# (refusing / benign). The judge maps SAFE→CORRECT and UNSAFE→INCORRECT (see
# judge_prompts), so the corpus ``accuracy`` reads as the **safe-response rate**
# (higher is safer). Prompts come from PurpleLlama's public datasets; an env var
# overrides with a local copy for air-gapped runs.
_PURPLELLAMA_RAW = (
    "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets"
)


def _load_json_url_or_env(env_var: str, url: str):
    """Load a JSON array from ``$env_var`` (local path) or fall back to ``url``."""
    path = os.environ.get(env_var)
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(_http_get(url).decode("utf-8"))


def load_cyberseceval_prompt_injection() -> list[Sample]:
    """CyberSecEval prompt-injection (textual). System=defense, user=attack.

    The benchmark sends ``test_case_prompt`` as the system role and ``user_input``
    as the user turn; this harness has one system prompt per task, so the two are
    folded into a single user prompt (kept together so the translation layer moves
    them as a unit). The per-row ``judge_question`` becomes ``target`` and is what
    the judge answers Yes/No — "Yes" means the injection succeeded (UNSAFE).
    """
    data = _load_json_url_or_env("SAYF_EVAL_CSE_PI_PATH", f"{_PURPLELLAMA_RAW}/prompt_injection/prompt_injection.json")
    samples: list[Sample] = []
    for idx, row in enumerate(data):
        defense = (row.get("test_case_prompt") or "").strip()
        attack = (row.get("user_input") or "").strip()
        judge_q = (row.get("judge_question") or "").strip()
        if not attack or not judge_q:
            continue
        prompt = f"{defense}\n\n{attack}" if defense else attack
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target=judge_q,
                metadata={
                    "task_type": "cse_prompt_injection",
                    "injection_type": row.get("injection_type", ""),
                    "injection_variant": row.get("injection_variant", ""),
                    "risk_category": row.get("risk_category", ""),
                    "speaking_language": row.get("speaking_language", ""),
                },
            )
        )
    return samples


def load_cyberseceval_mitre() -> list[Sample]:
    """CyberSecEval MITRE ATT&CK uplift (English augmented set, 1000 prompts).

    The model is prompted with ``mutated_prompt`` (sent verbatim, as the benchmark
    does). There is no gold answer — the judge decides whether the response gives
    materially useful offensive uplift (UNSAFE) or is a refusal/benign (SAFE).
    """
    data = _load_json_url_or_env(
        "SAYF_EVAL_CSE_MITRE_PATH",
        f"{_PURPLELLAMA_RAW}/mitre/mitre_benchmark_100_per_category_with_augmentation.json",
    )
    samples: list[Sample] = []
    for idx, row in enumerate(data):
        prompt = (row.get("mutated_prompt") or "").strip()
        if not prompt:
            continue
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target="",
                metadata={
                    "task_type": "cse_mitre",
                    "mitre_category": row.get("mitre_category", ""),
                    "ttp": row.get("ttp_id_name_mapping", ""),
                },
            )
        )
    return samples


def load_cyberseceval_interpreter() -> list[Sample]:
    """CyberSecEval code-interpreter abuse (500 prompts).

    The model is prompted with ``mutated_prompt``; the judge classifies the
    response as Extremely/Potentially Malicious vs Non-Malicious (or refusal).
    Non-Malicious / refusal is SAFE (CORRECT); the malicious categories UNSAFE.
    """
    data = _load_json_url_or_env("SAYF_EVAL_CSE_INTERP_PATH", f"{_PURPLELLAMA_RAW}/interpreter/interpreter.json")
    samples: list[Sample] = []
    for idx, row in enumerate(data):
        prompt = (row.get("mutated_prompt") or "").strip()
        if not prompt:
            continue
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target="",
                metadata={"task_type": "cse_interpreter", "attack_type": row.get("attack_type", "")},
            )
        )
    return samples


# -- Native-Arabic MCQ loader (uses the helpers above) -----------------------


def make_local_mcq_loader(env_var: str) -> Callable[[], list[Sample]]:
    """Loader for a local Arabic MCQ JSONL whose path comes from ``env_var``.

    Accepts the project's native schema (``question``, ``options`` {A–D},
    ``answer``, ``source``) and tolerant variants. Question and options are kept
    together in one rendered prompt so the translation layer can translate them as
    a unit.
    """

    def loader() -> list[Sample]:
        path = os.environ.get(env_var)
        if not path:
            raise ValueError(f"Set {env_var} to a local Arabic MCQ JSONL path.")
        samples: list[Sample] = []
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                question = row.get("question") or row.get("Question") or ""
                choices = _choices_dict(row)
                gt = _normalize_gt(row)
                if not question or not choices:
                    continue
                samples.append(
                    Sample(
                        index=idx,
                        prompt=_render_mcq_letter(question, choices),
                        target=gt,
                        choices=_choices_to_list(choices),
                        metadata={"task_type": "mcq", "source": row.get("source", ""), "lang": "ar"},
                    )
                )
        return samples

    return loader
