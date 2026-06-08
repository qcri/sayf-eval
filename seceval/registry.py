"""Task registry — per-benchmark configuration as data.

The registry maps a task name to a :class:`~seceval.task.Task`. Each entry binds
the unified prompt/system prompt, the calibrated token budget, a lazy dataset
loader (from ``datasets.py``), and the scorer kind. MVP entries (``mcq``,
``seceval``, ``vsp``, ``taa``) are added in Phase 2; the remaining ~20 tasks in
Phase 3. This module stays import-light: loaders are referenced, not invoked.
"""

from __future__ import annotations

from seceval.task import Task

# Populated as tasks are ported. Keep registrations in datasets-adjacent modules
# or register here directly via ``register(Task(...))``.
TASKS: dict[str, Task] = {}


def register(task: Task) -> Task:
    """Add a task to the registry (idempotent overwrite by name)."""
    TASKS[task.name] = task
    return task


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(
            f"Unknown task {name!r}. Registered: {sorted(TASKS)}"
        ) from None


def available_tasks() -> list[str]:
    return sorted(TASKS)
