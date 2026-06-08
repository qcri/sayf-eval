"""Task registrations. Importing this package populates the registry.

MVP subset (Phase 2): ``mcq``, ``seceval``, ``vsp``, ``taa``. Remaining ~20
tasks are added in Phase 3. Per-task token budgets follow the calibrated values
documented in the original pipeline README.
"""

from __future__ import annotations

from seceval import datasets as ds
from seceval.registry import register
from seceval.task import Task

# SecEval system instruction (ported from collect_seceval).
_SECEVAL_INSTRUCTION = (
    "Below are multiple-choice questions concerning cybersecurity. Please select "
    "the correct answers and respond with the letters ABCD only."
)

# CTI-Bench tasks carry a domain system prompt in the original harness.
_CTI_SYSTEM = "You are a cybersecurity expert specializing in cyberthreat intelligence."

register(Task(
    name="mcq",
    task_type="mcq",
    loader=ds.load_cti_mcq,
    system_prompt=_CTI_SYSTEM,
    max_tokens=1024,
))

register(Task(
    name="seceval",
    task_type="seceval",
    loader=ds.load_seceval,
    system_prompt=_SECEVAL_INSTRUCTION,
    max_tokens=256,
))

register(Task(
    name="vsp",
    task_type="vsp",
    loader=ds.load_cti_vsp,
    system_prompt=_CTI_SYSTEM,
    max_tokens=2048,
))

register(Task(
    name="taa",
    task_type="taa",
    loader=ds.load_cti_taa,
    system_prompt=_CTI_SYSTEM,
    max_tokens=512,
))
