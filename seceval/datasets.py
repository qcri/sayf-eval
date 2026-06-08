"""Dataset loaders — produce ``list[Sample]`` per benchmark.

Each loader normalizes a benchmark's raw rows into :class:`~seceval.task.Sample`
(the full rendered prompt text + ground truth + choices). Prompt construction is
ported from the original ``collect_*`` functions so task semantics are unchanged.
Loaders are lazy (called only when a task runs), so importing the registry does
not trigger downloads.
"""

from __future__ import annotations

import csv
import io
import urllib.request

from seceval.task import Sample

# MCQ instruction wording, ported from collect_huggingface_benchmark.
_MCQ_INSTRUCTION = (
    "You are given multiple choice questions. Answer with the option letter "
    "(A, B, C, D) from the given choices directly."
)


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        return r.read()


def _render_mcq(question: str, choices) -> tuple[str, list[str] | None]:
    """Build an MCQ prompt body + a normalized choices list for the judge."""
    prompt = _MCQ_INSTRUCTION + "\n\nQuestion: " + question + "\n"
    rendered: list[str] = []
    if isinstance(choices, dict):
        for key in ["A", "B", "C", "D"]:
            if key in choices:
                prompt += f"{key}. {choices[key]}\n"
                rendered.append(f"{key}. {choices[key]}")
    elif isinstance(choices, list):
        letters = ["A", "B", "C", "D"]
        for i, opt in enumerate(choices[:4]):
            prompt += f"{letters[i]}. {opt}\n"
            rendered.append(f"{letters[i]}. {opt}")
    prompt += "Answer:"
    return prompt, (rendered or None)


# -- CTI-Bench MCQ (RISys-Lab HF) -------------------------------------------

def load_cti_mcq() -> list[Sample]:
    from datasets import load_dataset

    ds = load_dataset("RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-mcq", split="test")
    samples: list[Sample] = []
    for idx, row in enumerate(ds):
        question = row.get("Prompt") or row.get("prompt") or row.get("question")
        if not question:
            continue
        gt = row.get("GT") or row.get("solution") or row.get("answer") or ""
        choices = row.get("answers") or row.get("choices") or row.get("options")
        if choices:
            prompt, rendered = _render_mcq(question, choices)
        else:
            prompt, rendered = question, None
        samples.append(
            Sample(
                index=idx,
                prompt=prompt,
                target=str(gt).strip(),
                choices=rendered,
                metadata={"task_type": "mcq", "subset": "cti-mcq"},
            )
        )
    return samples


# -- CTI-Bench VSP (RISys-Lab HF) -------------------------------------------

def load_cti_vsp() -> list[Sample]:
    from datasets import load_dataset

    ds = load_dataset("RISys-Lab/Benchmarks_CyberSec_CTI-Bench", "cti-vsp", split="test")
    samples: list[Sample] = []
    for idx, row in enumerate(ds):
        # VSP prompts are already fully formatted by the benchmark.
        question = row.get("Prompt") or row.get("prompt") or row.get("question")
        if not question:
            continue
        gt = row.get("GT") or row.get("solution") or ""
        samples.append(
            Sample(
                index=idx,
                prompt=question,
                target=str(gt).strip(),
                metadata={"task_type": "vsp", "subset": "cti-vsp"},
            )
        )
    return samples


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
    import json

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
