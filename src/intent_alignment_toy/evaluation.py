"""Evaluation loops and strict action-output contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from intent_alignment_toy.environment import Task, TextNavigationEnv

ACTION_PATTERN = re.compile(r"\s*(UP|DOWN|LEFT|RIGHT)\s*")
SYSTEM_PROMPT = "你是一个严格遵守动作接口的文字环境智能体。"


def parse_action(raw_output: str) -> str | None:
    match = ACTION_PATTERN.fullmatch(raw_output)
    return match.group(1) if match else None


def build_policy_prompt(task: Task, env: TextNavigationEnv, actions: list[str]) -> str:
    """Legacy v1.1 full-state prompt retained for historical reproducibility."""
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


def build_initial_agent_prompt(task: Task) -> str:
    """Initial user turn for the v2 stateful agent-loop contract."""
    env = TextNavigationEnv(task)
    return f"""你正在操作一个二维文字移动环境。

规则：
- 起点是 (0, 0)。
- 每次动作恰好移动 1 米。
- 动作与坐标变化：UP=(0,+1)，DOWN=(0,-1)，LEFT=(-1,0)，RIGHT=(+1,0)。
- 可用动作只有 UP、DOWN、LEFT、RIGHT。
- 每一轮只能选择一个动作；环境执行后会返回新观察。
- 到达目标时环境自动结束。
- 你必须在给定动作预算内完成任务。

用户意图：{task.instruction}
环境观察：{env.observation()}

只输出下一步的一个动作标签，不要输出解释、标点或代码块。"""


def build_agent_feedback(env: TextNavigationEnv, actions: list[str]) -> str:
    """Environment-owned observation appended between policy action turns."""
    action_history = "、".join(actions) if actions else "无"
    return f"""环境已执行上一动作。
环境观察：{env.observation()}
已执行动作：{action_history}
只输出下一步的一个动作标签，不要输出解释、标点或代码块。"""


def build_agent_messages(
    task: Task, actions: list[str], raw_outputs: list[str] | None = None
) -> list[dict[str, str]]:
    """Build the semantic full conversation used for greedy evaluation.

    ``actions`` contains normalized environment actions. ``raw_outputs`` preserves
    the exact assistant text that produced each accepted action, matching the
    native veRL loop's history. Passing no raw outputs is a convenience for
    deterministic fixtures whose assistant text is identical to the action label.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_agent_prompt(task)},
    ]
    replay_env = TextNavigationEnv(task)
    replay_actions: list[str] = []
    assistant_outputs = list(actions) if raw_outputs is None else list(raw_outputs)
    if len(assistant_outputs) != len(actions):
        raise ValueError("raw output history must align with normalized actions")
    for action, raw_output in zip(actions, assistant_outputs):
        if replay_env.terminated:
            raise ValueError("action history continues after environment termination")
        if parse_action(raw_output) != action:
            raise ValueError("raw output does not match its normalized action")
        messages.append({"role": "assistant", "content": raw_output})
        replay_actions.append(action)
        replay_env.step(action)
        if not replay_env.terminated:
            messages.append(
                {
                    "role": "user",
                    "content": build_agent_feedback(replay_env, replay_actions),
                }
            )
    return messages


def evaluate_with_policy(
    tasks: list[Task],
    policy: Callable[
        [list[tuple[Task, TextNavigationEnv, list[str], list[str]]]], list[str]
    ],
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("evaluation requires at least one task")
    episodes = [
        {"task": task, "env": TextNavigationEnv(task), "actions": [], "raw_outputs": []}
        for task in tasks
    ]

    while True:
        active = [episode for episode in episodes if not episode["env"].terminated]
        if not active:
            break
        requests = [
            (
                episode["task"],
                episode["env"],
                list(episode["actions"]),
                list(episode["raw_outputs"]),
            )
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
    threshold_successes = math.ceil(0.95 * total)
    return {
        "summary": {
            "successes": successes,
            "total": total,
            "success_rate": successes / total,
            "at_least_95_percent_requires": threshold_successes,
            "meets_at_least_95_percent": successes >= threshold_successes,
            "meets_strictly_above_95_percent": successes / total > 0.95,
        },
        "tasks": task_results,
    }
