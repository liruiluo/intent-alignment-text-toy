"""Load and validate the frozen intent-alignment pilot dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from intent_alignment_toy.environment import Task


def load_tasks(path: str | Path) -> list[Task]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    tasks = [
        Task(
            task_id=record["id"],
            instruction=record["instruction"],
            target=tuple(record["target"]),
            max_steps=record["max_steps"],
        )
        for record in records
    ]
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[Task]) -> None:
    if len(tasks) != 20:
        raise ValueError(f"expected exactly 20 tasks, got {len(tasks)}")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")

    minimal_lengths = Counter()
    for task in tasks:
        target_x, target_y = task.target
        if (target_x == 0) == (target_y == 0):
            raise ValueError(f"{task.task_id}: target must lie on one cardinal axis")
        distance = abs(target_x) + abs(target_y)
        if not 1 <= distance <= 5:
            raise ValueError(f"{task.task_id}: target distance must be in [1, 5]")
        if not distance <= task.max_steps <= 5:
            raise ValueError(f"{task.task_id}: invalid action budget")
        minimal_lengths[distance] += 1

    if minimal_lengths != Counter({length: 4 for length in range(1, 6)}):
        raise ValueError(
            f"expected four tasks at each distance 1-5, got {minimal_lengths}"
        )
