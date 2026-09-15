"""Deterministic two-dimensional text navigation environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

ACTION_DELTAS: Final[dict[str, tuple[int, int]]] = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    target: tuple[int, int]
    max_steps: int


@dataclass(frozen=True)
class StepResult:
    position: tuple[int, int]
    steps_used: int
    remaining_steps: int
    terminated: bool
    success: bool
    reason: str
    trajectory: tuple[tuple[int, int], ...]


class TextNavigationEnv:
    """Execute one-metre cardinal moves under a bounded action budget."""

    def __init__(self, task: Task):
        if not 1 <= task.max_steps <= 5:
            raise ValueError("max_steps must be between 1 and 5")
        self.task = task
        self.position = (0, 0)
        self.steps_used = 0
        self.terminated = False
        self.success = False
        self.reason = "running"
        self.trajectory = [(0, 0)]

    @property
    def remaining_steps(self) -> int:
        return self.task.max_steps - self.steps_used

    def observation(self) -> str:
        history_state = "已到达目标" if self.success else "尚未到达目标"
        return (
            f"当前位置：({self.position[0]}, {self.position[1]})；"
            f"剩余动作数：{self.remaining_steps}；{history_state}。"
        )

    def step(self, action: str) -> StepResult:
        if self.terminated:
            raise RuntimeError("episode has already terminated")
        if action not in ACTION_DELTAS:
            self.terminated = True
            self.reason = "invalid_action"
            return self.result()

        delta_x, delta_y = ACTION_DELTAS[action]
        self.position = (
            self.position[0] + delta_x,
            self.position[1] + delta_y,
        )
        self.steps_used += 1
        self.trajectory.append(self.position)

        if self.position == self.task.target:
            self.success = True
            self.terminated = True
            self.reason = "target_reached"
        elif self.steps_used == self.task.max_steps:
            self.terminated = True
            self.reason = "budget_exhausted"

        return self.result()

    def result(self) -> StepResult:
        return StepResult(
            position=self.position,
            steps_used=self.steps_used,
            remaining_steps=self.remaining_steps,
            terminated=self.terminated,
            success=self.success,
            reason=self.reason,
            trajectory=tuple(self.trajectory),
        )
