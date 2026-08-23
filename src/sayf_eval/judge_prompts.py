"""Unified LLM-as-judge prompt: extract + verdict in one call.

Ported verbatim from the original harness
(``unified-benchmark-pipeline/run_evaluate_llm_judge.py::create_judge_prompt``,
lines 86-254). The judge does extraction AND verdict; comparison rules are
described in natural language per task. No regex/Python post-processing of the
model answer — the judge returns the canonical form.

Unified output schema (every task)::

    {
      "extracted_answer": "<verbatim model answer in canonical form>",
      "verdict": "CORRECT" | "INCORRECT",
      "justification": "<one sentence>"
    }
"""

from __future__ import annotations


# task_type -> (format_hint, compare_rule). MCQ-family types share one entry.
# Only task_type values that actually reach the judge are listed. Most MCQ
# benchmarks (cybermetric, cissp, mmlu-cs, secbench, redsage_*) register with
# task_type="mcq"; ``secure`` and ``ckt`` keep a distinct task_type but are
# judged with the same MCQ rule.
_MCQ_TYPES = ("mcq", "secure", "ckt")

_MCQ_RULE = (
    "a single uppercase letter (A, B, C, D, or E)",
    "Verdict is CORRECT if the extracted letter equals the correct answer (case-insensitive). Otherwise INCORRECT.",
)

_RULES: dict[str, tuple[str, str]] = {
    "secure_kcv": (
        # True/False verification task (T/F/X), NOT A-E multiple choice.
        # X is a first-class answer the task offers ("if you do not know, return
        # X"), so a deliberate "unknown" must extract to X. A response with no
        # discernible answer must fall back to the NONE sentinel (never a valid
        # label) so abstention stays INCORRECT — do NOT default it to X, which
        # would score unanswered items as correct whenever the gold answer is X.
        "a single uppercase letter: T (true) or F (false); or X when the model "
        "explicitly answers that it does not know / cannot be determined. This "
        "is a True/False verification task, not multiple choice. If the response "
        'contains no discernible answer at all, output "NONE".',
        "Verdict is CORRECT if the extracted letter (T, F, or X) equals the "
        'correct answer (case-insensitive). "NONE" is never a valid answer and '
        "is always INCORRECT.",
    ),
    "seceval": (
        "uppercase letters concatenated and sorted alphabetically, e.g. "
        '"B" or "AB" or "ACD". If the model gave no clear answer, output "NONE".',
        "Verdict is CORRECT only if the extracted set of letters equals the correct "
        "set exactly (case-insensitive, order ignored). Partial matches are INCORRECT: "
        'e.g. correct="ABC" → "ABC" CORRECT; "AB" INCORRECT (missing); "ABCD" INCORRECT (extra).',
    ),
    "rcm": (
        'comma-separated sorted CWE IDs in canonical form "CWE-NNN", '
        'e.g. "CWE-79" or "CWE-79,CWE-89". If no valid CWE was given, output "NONE".',
        "Verdict is CORRECT only if the extracted CWE-ID set equals the correct CWE-ID set exactly "
        '(treat "CWE-79", "cwe-79", "79" as equivalent — same numeric ID). '
        "Partial/missing/extra IDs are INCORRECT.",
    ),
    "vsp": (
        'a CVSS v3.1 vector string starting with "CVSS:3.1/", '
        'e.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H". '
        'If the model started with "CVSS:3.0/" change it to "CVSS:3.1/". '
        'If no vector prefix is present, prepend "CVSS:3.1/". '
        'If no valid CVSS vector is found, output "NONE".',
        "Verdict is CORRECT if the extracted vector equals the correct vector after "
        "case-insensitive normalization and prefix normalization (CVSS:3.0/ ≡ CVSS:3.1/). "
        "Otherwise INCORRECT.",
    ),
    "ate": (
        'comma-separated sorted PARENT MITRE ATT&CK technique IDs, e.g. "T1027,T1059". '
        'IMPORTANT: strip any subtechnique suffix — "T1059.001" must become "T1059". '
        "Output ONLY parent IDs (uppercase T followed by 4 digits, no dot). "
        'If no valid technique was given, output "NONE".',
        "Verdict is CORRECT only if the extracted set of PARENT technique IDs equals the "
        "correct set of parent IDs exactly. Always strip subtechnique suffixes from BOTH "
        'extracted and ground truth before comparing ("T1059.001" → "T1059"). '
        "Partial/missing/extra parent IDs are INCORRECT.",
    ),
    "rms": (
        'comma-separated sorted MITRE mitigation IDs in form "MNNNN", '
        'e.g. "M1018,M1026,M1047". If no valid mitigation was given, output "NONE".',
        "Verdict is CORRECT only if the extracted set of mitigation IDs equals the correct "
        "set exactly (case-insensitive, order ignored). Partial/missing/extra IDs are INCORRECT.",
    ),
    "taa": (
        "the threat actor name as the model gave it, in canonical form (most common name). "
        'If no actor was given, output "NONE".',
        "Verdict is CORRECT if the extracted actor name refers to the same threat actor as "
        "the correct answer (case-insensitive; common aliases are equivalent, e.g. "
        '"APT28" ≡ "Fancy Bear", "Lazarus" ≡ "Lazarus Group"). Otherwise INCORRECT.',
    ),
    "sevenllm": (
        "a concise single-line summary (≤300 chars) of the FACTS the model extracted or "
        "generated. For information-extraction tasks (JSON output): list the key "
        'entities/values the model produced as "field: value; field: value" pairs '
        '(e.g., "Malware: Dridex; capabilities: payload drop, ransomware; evasion: '
        'password-protected Excel"). For analysis-generation tasks (text output): a '
        "one-sentence paraphrase of the model's main claim. If the model gave no usable "
        'answer (empty, refusal, garbage), output "NONE".',
        "This is a SEVENLLM cybersecurity information-extraction (JSON output) or "
        "analysis-generation (text output) task. The instruction shown to the model in the "
        "Question block specifies which category (e.g. Malware Feature Extraction, Threat "
        "Analysis, Time Element Acquisition). Compare the extracted answer to the correct "
        "answer SEMANTICALLY, not by schema or wording:\n"
        "- The model's JSON schema may legitimately differ from the ground-truth schema "
        "(different key names, different nesting). What matters is whether the same "
        "factual entities/values are captured.\n"
        '- Treat common aliases and abbreviations as equivalent (e.g. "US" ≡ "United '
        'States", "AES-256" ≡ "AES", semantically-similar phrasings).\n'
        "- Verdict is CORRECT if the model captures the CENTRAL facts of the correct "
        "answer (main entity, primary list, key claim) with no contradictions on those "
        "central facts and no major fabricated entities. Missing peripheral details are OK.\n"
        "- Verdict is INCORRECT if the model misses the central facts, contradicts the "
        "ground truth on a key entity/value, or hallucinates major entities not supported "
        "by the source.\n"
        "Be lenient on phrasing and schema, strict on factual coverage of the central facts.",
    ),
}

_SUPPORTED = ", ".join(list(_MCQ_TYPES) + sorted(_RULES))


def create_judge_prompt(
    task_type: str,
    question: str,
    model_answer: str,
    ground_truth: str,
    extra_context: dict | None = None,
) -> str:
    """Build the unified judge prompt for ``task_type``.

    ``extra_context["choices"]`` (MCQ family) is appended to the question block.
    Raises ``ValueError`` for an unknown task type.
    """
    if task_type in _MCQ_TYPES:
        format_hint, compare_rule = _MCQ_RULE
        ctx = ""
        if extra_context and "choices" in extra_context:
            ctx = f"\n\nAnswer Choices:\n{extra_context['choices']}"
        question_block = f"Question:\n{question}{ctx}"
    elif task_type in _RULES:
        format_hint, compare_rule = _RULES[task_type]
        question_block = f"Question:\n{question}"
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Supported: {_SUPPORTED}")

    return f"""You are a strict evaluator for a cybersecurity benchmark. Your role has TWO steps.

{question_block}

Correct Answer: {ground_truth}

Model's Response:
{model_answer}

STEP 1 — EXTRACT the model's answer.
- Look for the model's explicit final answer (e.g., "Answer:", "Final answer:", concluding line, single bolded line at top).
- IGNORE letters/IDs/text that appear only inside explanations of other options or thinking-aloud prose.
- If the model gives multiple inconsistent answers, use its most prominent/final selection.
- DO NOT correct, infer, or improve the model's answer — extract verbatim what it actually said.
- Format the extracted answer as: {format_hint}

STEP 2 — VERDICT.
{compare_rule}

Output ONLY this JSON, nothing else:
{{
  "extracted_answer": "<your extraction in the format above>",
  "verdict": "CORRECT" or "INCORRECT",
  "justification": "<one short sentence explaining what you extracted and why the verdict>"
}}

Your judgment:"""
