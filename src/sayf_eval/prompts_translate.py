"""Translation prompts — lifted verbatim from the seed-mini translation pipeline.

These are the exact system prompts the project used to build its translated
benchmark sets, kept here so the in-harness translation layer reproduces the same
behavior rather than inventing new wording. Provenance is noted per block.

Sources (repo ``sayf-multilingual/seed_mini``):
* ``trans_bench_mcq.py``  — model-aware EN→AR MCQ translation (lines ~163-205)
* ``trans_bench_mcq.py``  — generic EN→AR prompt translation (line ~276)
* ``translate_ar_bench_en.py`` — AR→EN back-translation contract (lines ~65-82)
"""

from __future__ import annotations


# ── EN → AR, MCQ family (model-aware) ───────────────────────────────────────
# Verbatim from trans_bench_mcq.py. The key is matched as a substring against the
# translator model name (as the original did: ``if "fanar" in model.lower()``).
MCQ_EN2AR: dict[str, str] = {
    "fanar": (
        "You are a professional Arabic translator specialized in cybersecurity exams.\n"
        "Translate the following English MCQ and its four options into Modern Standard Arabic (MSA).\n"
        "Output: 1 line for question + 4 lines (أ:, ب:, ج:, د:). No numbering, no explanation."
    ),
    "llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a professional technical translator.\n"
        "Translate the following cybersecurity multiple-choice question (MCQ) and its four options "
        "into clear Modern Standard Arabic (MSA).\n"
        "Keep exactly this structure:\n"
        "1 Arabic sentence for the question.\n"
        "Then four lines beginning with A:, B:, C:, and D: — each line the Arabic translation of the option.\n"
        "Do not repeat the question in English. Do not add labels like 'السؤال' or 'الخيارات'.\n"
        "Preserve English acronyms such as DNS, HTTPS, TLS, CPU.\n"
        "Write complete Arabic sentences; avoid truncation or mixed fragments.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    ),
    "gpt-oss": (
        "<|start|>system<|message|>You are a professional Arabic cybersecurity translator.\n"
        "<|start|>user<|message|>\n"
        "Translate this English MCQ (question + four options) into Arabic (MSA). "
        "Ignore the answer key and avoid commentary.\n"
        "Output must be 1 Arabic line for the question and 4 Arabic lines for options (A–D)."
    ),
    "default": (
        "Translate the English MCQ (question + four options) into Arabic (MSA). "
        "Do not solve or explain. Keep the same structure (A–D)."
    ),
}

# Appended when RETRY_MODE is requested (verbatim from trans_bench_mcq.py).
MCQ_EN2AR_RETRY = (
    "\n\n[Retry]: Translate the following cybersecurity MCQ again into clear Modern Standard Arabic (MSA). "
    "Ensure the question and all four options are properly translated. "
    "Keep English acronyms like DNS, HTTPS, TLS, and CPU."
)


# ── EN → AR, generic (any rendered prompt / open-ended task) ────────────────
# Verbatim from trans_bench_mcq.py (the open-ended branch). Robust for prompts
# that are not strict 4-option MCQs (RCM/VSP/TAA/SEvenLLM/SECURE).
GENERIC_EN2AR = (
    "Translate the following cybersecurity prompt into Modern Standard Arabic (MSA). "
    "Preserve technical terms (DNS, TLS, etc.), no commentary."
)


# ── AR → EN, MCQ family (translate-test back-translation) ───────────────────
# Verbatim from translate_ar_bench_en.py. Used to render native-Arabic benchmarks
# in English (the translate-test baseline).
MCQ_AR2EN = (
    "You MUST follow this structured-output contract:\n"
    "1) Output NOTHING outside the <final>...</final> block.\n"
    "2) Inside <final>, output valid JSON with keys exactly:\n"
    '   - "question_en": string\n'
    '   - "choices_en": array of 4 strings\n'
    "3) Task: Translate the Arabic question and its four Arabic options into English.\n"
    "4) IMPORTANT:\n"
    "   - Translate question and options TOGETHER to preserve meaning.\n"
    "   - Keep cybersecurity terminology accurate.\n"
    "   - Do NOT add extra commentary.\n"
    "   - Do NOT include the Arabic text in the output.\n"
    "5) Output format requirements:\n"
    "   - choices_en must be exactly 4 items.\n"
    "   - Each item must start with 'A: ', 'B: ', 'C: ', 'D: ' in order.\n"
    "6) If the Arabic options are labeled using Arabic letters (أ/ب/ج/د), map them in the same order to A/B/C/D.\n"
)

# Generic AR→EN for open-ended prompts (no structured-output contract).
GENERIC_AR2EN = (
    "Translate the following cybersecurity prompt into clear English. "
    "Preserve technical terms (DNS, TLS, etc.), no commentary."
)


def select_en2ar(model_name: str, mcq: bool, retry: bool = False, model_aware: bool = False) -> str:
    """Pick the EN→AR system prompt.

    Defaults to the chat-safe prompts (MCQ ``default`` / generic). The
    ``fanar``/``llama``/``gpt-oss`` variants from trans_bench_mcq embed raw
    chat-template tokens meant for a non-chat serving path, so they are opt-in via
    ``model_aware=True`` (mirroring the original model dispatch).
    """
    if not mcq:
        return GENERIC_EN2AR
    key = "default"
    if model_aware:
        m = (model_name or "").lower()
        if "fanar" in m:
            key = "fanar"
        elif "llama" in m:
            key = "llama"
        elif "gpt-oss" in m or "gptoss" in m:
            key = "gpt-oss"
    prompt = MCQ_EN2AR[key]
    return prompt + MCQ_EN2AR_RETRY if retry else prompt


def select_ar2en(mcq: bool) -> str:
    """Pick the AR→EN system prompt for the translate-test direction."""
    return MCQ_AR2EN if mcq else GENERIC_AR2EN
