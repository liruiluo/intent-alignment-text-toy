"""Evaluation loop and strict action-output contract."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from intent_alignment_toy.environment import Task, TextNavigationEnv

ACTION_PATTERN = re.compile(r"\s*(UP|DOWN|LEFT|RIGHT)\s*")


def parse_action(raw_output: str) -> str | None:
    match = ACTION_PATTERN.fullmatch(raw_output)
    return match.group(1) if match else None


def build_policy_prompt(task: Task, env: TextNavigationEnv, actions: list[str]) -> str:
    action_history = "、".join(actions) if actions else "无"
    return f"""你正在操作一个二维文字移动环境。

规则：
- 起点是 (0, 0)。
- 每次动作恰好移动 1 米。
- 动作与坐标变化：UP=(0,+1)，DOWN=(0,-1)，LEFT=(-1,0)，RIGHT=(+1,0)。
- 可用动作只有 UP、DOWN、LEFT、RIGHT。
- 环境到达目标后会自动结束。
- 你必须在给定动作预算内完成任务。

用户意图：{task.instruction}
{env.observation()}
此前动作：{action_history}

只输出下一步的一个动作标签，不要输出解释、标点或代码块。"""


def evaluate_with_policy(
    tasks: list[Task],
    policy: Callable[[list[tuple[Task, TextNavigationEnv, list[str]]]], list[str]],
) -> dict[str, Any]:
    episodes = [
        {"task": task, "env": TextNavigationEnv(task), "actions": [], "raw_outputs": []}
        for task in tasks
    ]

    while True:
        active = [episode for episode in episodes if not episode["env"].terminated]
        if not active:
            break
        requests = [
            (episode["task"], episode["env"], list(episode["actions"]))
            for episode in active
        ]
        raw_outputs = policy(requests)
        if len(raw_outputs) != len(active):
            raise ValueError("policy must return one output per active episode")

        for episode, raw_output in zip(active, raw_outputs):
            action = parse_action(raw_output)
            episode["raw_outputs"].append(raw_output)
            if action is None:
                episode["env"].step("INVALID")
            else:
                episode["actions"].append(action)
                episode["env"].step(action)

    task_results = []
    for episode in episodes:
        task = episode["task"]
        result = episode["env"].result()
        task_results.append(
            {
                "id": task.task_id,
                "instruction": task.instruction,
                "target": list(task.target),
                "max_steps": task.max_steps,
                "actions": episode["actions"],
                "raw_outputs": episode["raw_outputs"],
                **asdict(result),
            }
        )

    successes = sum(result["success"] for result in task_results)
    total = len(task_results)
    return {
        "summary": {
            "successes": successes,
            "total": total,
            "success_rate": successes / total,
            "meets_at_least_95_percent": successes / total >= 0.95,
            "meets_strictly_above_95_percent": successes / total > 0.95,
        },
        "tasks": task_results,
    }
