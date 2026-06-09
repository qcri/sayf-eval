"""Scorer — LLM-as-judge extraction + verdict (sample level).

The judge is another :class:`~sayf_eval.model.Model`. For each sample the scorer
builds the unified judge prompt, calls the judge, and parses a strict-JSON
verdict. Two invariants ported from the original harness:

- **Reasoning handling:** a ``<think>…</think>`` block is stripped from the model
  answer before judging (backend-agnostic; the RedSage/Qwen3 fix). Optional
  ``stop`` sequences are then applied to the answer portion only.
- **Skipped semantics:** a judge-API failure (empty / ``ERROR:`` / content
  filter) yields ``skipped=True`` so the corpus metric excludes the sample from
  *both* numerator and denominator. An unparseable *model* answer is still
  judged and counts as incorrect (denominator policy lives in ``metrics.py``).

Parsing logic mirrors ``run_evaluate_llm_judge.py::parse_judge_response``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sayf_eval.judge_prompts import create_judge_prompt
from sayf_eval.model import GenParams, Model, Response


# JSON mode for the judge — strict object output.
JUDGE_RESPONSE_FORMAT = {"type": "json_object"}

_FAILURE_MARKERS = ("content management policy", "content_filter")


def strip_reasoning(text: str, stop: list[str] | None = None) -> str:
    """Strip a ``<think>`` block, then apply ``stop`` to the answer portion only.

    This is the backend-agnostic version of the harness's thinking-model
    handling: reasoning preambles must not be fed to the judge, and a stop
    sequence (e.g. RedSage's ``"\\n"``) must fire on the *answer*, not inside the
    discarded thinking trace.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if stop:
        for s in stop:
            if s and s in text:
                text = text.split(s)[0]
    return text.strip()


@dataclass
class SampleVerdict:
    """Per-sample judge result."""

    extracted_answer: str
    verdict: str  # "CORRECT" | "INCORRECT" | ""
    is_correct: bool
    skipped: bool
    justification: str
    judge_response: str

    def to_dict(self) -> dict:
        return {
            "extracted_answer": self.extracted_answer,
            "verdict": self.verdict,
            "is_correct": self.is_correct,
            "skipped": self.skipped,
            "justification": self.justification,
            "judge_response": self.judge_response,
        }


def parse_judge_response(judge_response: str, task_type: str) -> SampleVerdict:
    """Parse a raw judge string into a :class:`SampleVerdict`.

    Detects judge failures (skipped), parses the JSON object (markdown-fenced or
    bare), and falls back to regex extraction of ``verdict`` /
    ``extracted_answer`` if JSON parsing fails.
    """
    stripped = judge_response.strip()
    if not stripped or stripped.startswith("ERROR:") or any(m in judge_response for m in _FAILURE_MARKERS):
        return SampleVerdict(
            extracted_answer="",
            verdict="",
            is_correct=False,
            skipped=True,
            justification="Judge API failure — sample excluded from scoring",
            judge_response=judge_response,
        )

    json_str: str | None = None
    try:
        m = re.search(r"```(?:json)?\s*({.*?})\s*```", judge_response, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            m = re.search(r"{.*}", judge_response, re.DOTALL)
            json_str = m.group(0) if m else judge_response
        parsed = json.loads(json_str)
        extracted = str(parsed.get("extracted_answer", "")).strip()
        verdict = str(parsed.get("verdict", "")).strip().upper()
        justification = str(parsed.get("justification", "No justification provided"))
    except (json.JSONDecodeError, AttributeError):
        source = json_str if json_str else judge_response
        vm = re.search(r'"verdict"\s*:\s*"(CORRECT|INCORRECT)"', source, re.IGNORECASE)
        em = re.search(r'"extracted_answer"\s*:\s*"([^"]*)"', source)
        jm = re.search(r'"justification"\s*:\s*"([^"]*)"', source)
        verdict = vm.group(1).upper() if vm else ""
        extracted = em.group(1) if em else ""
        justification = jm.group(1) if jm else "Fallback parsing - JSON parse failed"

    return SampleVerdict(
        extracted_answer=extracted,
        verdict=verdict,
        is_correct=(verdict == "CORRECT"),
        skipped=False,
        justification=justification,
        judge_response=judge_response,
    )


def format_choices(choices) -> str | None:
    """Render choices (dict or list) into the judge prompt's choices block."""
    if choices is None:
        return None
    if isinstance(choices, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
    if isinstance(choices, list):
        return "\n".join(str(c) for c in choices)
    return str(choices)


class JudgeScorer:
    """Runs the judge over collected model responses (sample level).

    Args:
        judge: the judge :class:`~sayf_eval.model.Model`.
        judge_params: generation params for the judge. ``response_format``
            defaults to JSON mode if unset.
    """

    def __init__(self, judge: Model, judge_params: GenParams | None = None) -> None:
        self.judge = judge
        self.params = judge_params or GenParams(max_tokens=512)
        if self.params.response_format is None:
            self.params.response_format = JUDGE_RESPONSE_FORMAT

    def build_prompt(
        self,
        task_type: str,
        question: str,
        model_answer: str,
        target: str,
        choices=None,
        answer_stop: list[str] | None = None,
    ) -> str:
        answer = strip_reasoning(model_answer, stop=answer_stop)
        extra = {}
        rendered = format_choices(choices)
        if rendered is not None:
            extra["choices"] = rendered
        return create_judge_prompt(task_type, question, answer, target, extra)

    def score_sample(
        self,
        task_type: str,
        question: str,
        model_answer: str,
        target: str,
        choices=None,
        answer_stop: list[str] | None = None,
    ) -> SampleVerdict:
        """Judge one sample. Never raises on judge failure (returns skipped)."""
        prompt = self.build_prompt(task_type, question, model_answer, target, choices, answer_stop)
        resp: Response = self.judge.generate([{"role": "user", "content": prompt}], self.params)
        if not resp.ok:
            # Surface as an explicit failure marker the parser maps to skipped.
            return parse_judge_response("ERROR: judge call not ok", task_type)
        return parse_judge_response(resp.text, task_type)

    def score_batch(self, items: list[dict]) -> list[SampleVerdict]:
        """Judge many samples concurrently.

        Each item: ``{task_type, question, model_answer, target, choices?,
        answer_stop?}``. Order is preserved.
        """
        prompts = [
            [
                {
                    "role": "user",
                    "content": self.build_prompt(
                        it["task_type"],
                        it["question"],
                        it["model_answer"],
                        it.get("target", ""),
                        it.get("choices"),
                        it.get("answer_stop"),
                    ),
                }
            ]
            for it in items
        ]
        responses = self.judge.generate_batch(prompts, self.params)
        out: list[SampleVerdict] = []
        for it, resp in zip(items, responses):
            text = resp.text if resp.ok else "ERROR: judge call not ok"
            out.append(parse_judge_response(text, it["task_type"]))
        return out
