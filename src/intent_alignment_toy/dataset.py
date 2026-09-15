"""Load and validate intent-alignment task datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from intent_alignment_toy.environment import ACTION_DELTAS, Task

_DIRECTION_FROM_TARGET = {
    (1, 0): "RIGHT",
    (-1, 0): "LEFT",
    (0, 1): "UP",
    (0, -1): "DOWN",
}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records without discarding dataset metadata."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tasks(path: str | Path, *, expected_count: int | None = None) -> list[Task]:
    """Load task objects from either the legacy pilot or a generated split."""
    records = load_records(path)
    validate_task_records(records, expected_count=expected_count)
    tasks = [
        Task(
            task_id=record["id"],
            instruction=record["instruction"],
            target=tuple(record["target"]),
            max_steps=record["max_steps"],
        )
        for record in records
    ]
    validate_tasks(tasks, expected_count=expected_count)
    return tasks


def target_direction(target: tuple[int, int]) -> str:
    """Return the cardinal action that moves from the origin toward ``target``."""
    target_x, target_y = target
    distance = abs(target_x) + abs(target_y)
    if distance == 0 or (target_x == 0) == (target_y == 0):
        raise ValueError("target must lie on exactly one non-zero cardinal axis")
    unit = (target_x // distance, target_y // distance)
    return _DIRECTION_FROM_TARGET[unit]


def validate_tasks(tasks: list[Task], *, expected_count: int | None = None) -> None:
    """Validate the environment-level contract shared by every dataset size."""
    if not tasks:
        raise ValueError("dataset must contain at least one task")
    if expected_count is not None and len(tasks) != expected_count:
        raise ValueError(f"expected exactly {expected_count} tasks, got {len(tasks)}")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")

    for task in tasks:
        target_direction(task.target)
        distance = abs(task.target[0]) + abs(task.target[1])
        if not 1 <= distance <= 5:
            raise ValueError(f"{task.task_id}: target distance must be in [1, 5]")
        if not distance <= task.max_steps <= 5:
            raise ValueError(f"{task.task_id}: invalid action budget")
        if not task.instruction.strip():
            raise ValueError(f"{task.task_id}: instruction must not be empty")


def validate_task_records(
    records: list[dict[str, Any]], *, expected_count: int | None = None
) -> None:
    """Validate JSONL records, including optional generated-split metadata."""
    required = {"id", "instruction", "target", "max_steps"}
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} records, got {len(records)}"
        )

    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record {index}: missing fields {sorted(missing)}")
        target = tuple(record["target"])
        direction = target_direction(target)
        distance = abs(target[0]) + abs(target[1])
        if "direction" in record and record["direction"] != direction:
            raise ValueError(
                f"{record['id']}: direction metadata does not match target"
            )
        if "distance" in record and record["distance"] != distance:
            raise ValueError(f"{record['id']}: distance metadata does not match target")


def validate_split_pair(
    train_records: list[dict[str, Any]], test_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate the frozen 400/100 lexical-family-disjoint split."""
    validate_task_records(train_records, expected_count=400)
    validate_task_records(test_records, expected_count=100)

    all_records = train_records + test_records
    ids = [record["id"] for record in all_records]
    instructions = [record["instruction"] for record in all_records]
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be unique across train and test")
    if len(set(instructions)) != len(instructions):
        raise ValueError("instructions must be unique across train and test")

    def validate_split(
        records: list[dict[str, Any]], split: str, expected_per_cell: int
    ) -> tuple[set[str], Counter[tuple[str, int]]]:
        families: set[str] = set()
        cells: Counter[tuple[str, int]] = Counter()
        for record in records:
            if record.get("split") != split:
                raise ValueError(f"{record['id']}: expected split={split!r}")
            family = record.get("template_family")
            if not isinstance(family, str) or not family:
                raise ValueError(f"{record['id']}: template_family is required")
            families.add(family)
            cells[(record["direction"], record["distance"])] += 1

        expected_cells = {
            (direction, distance)
            for direction in ACTION_DELTAS
            for distance in range(1, 6)
        }
        if set(cells) != expected_cells:
            raise ValueError(f"{split}: missing or extra semantic cells")
        wrong = {
            cell: count for cell, count in cells.items() if count != expected_per_cell
        }
        if wrong:
            raise ValueError(f"{split}: unexpected examples per semantic cell: {wrong}")
        return families, cells

    train_families, train_cells = validate_split(train_records, "train", 20)
    test_families, test_cells = validate_split(test_records, "test", 5)
    overlap = train_families & test_families
    if overlap:
        raise ValueError(f"train/test template families overlap: {sorted(overlap)}")
    if len(train_families) != 20 or len(test_families) != 5:
        raise ValueError("expected 20 train and 5 test template families")

    return {
        "train_count": len(train_records),
        "test_count": len(test_records),
        "train_template_families": sorted(train_families),
        "test_template_families": sorted(test_families),
        "train_cell_counts": {
            f"{direction}-{distance}": train_cells[(direction, distance)]
            for direction in ACTION_DELTAS
            for distance in range(1, 6)
        },
        "test_cell_counts": {
            f"{direction}-{distance}": test_cells[(direction, distance)]
            for direction in ACTION_DELTAS
            for distance in range(1, 6)
        },
    }
