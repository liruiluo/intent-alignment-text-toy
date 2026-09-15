"""Native veRL AgentLoop for the bounded text-navigation environment."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from intent_alignment_toy.environment import Task, TextNavigationEnv
from intent_alignment_toy.evaluation import build_agent_feedback, parse_action


class IntentAlignmentAgentLoop(AgentLoopBase):
    """Sample one atomic action per turn and execute it in ``TextNavigationEnv``."""

    def __init__(self, *args, action_max_new_tokens: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        if action_max_new_tokens < 1:
            raise ValueError("action_max_new_tokens must be positive")
        self.action_max_new_tokens = int(action_max_new_tokens)
        self.response_length = int(self.rollout_config.response_length)

    @staticmethod
    def _task_from_kwargs(kwargs: dict[str, Any]) -> Task:
        extra_info = kwargs.get("extra_info") or {}
        task_data = extra_info.get("task") or {}
        required = {"id", "instruction", "target", "max_steps"}
        missing = required - task_data.keys()
        if missing:
            raise ValueError(f"extra_info.task missing fields: {sorted(missing)}")
        return Task(
            task_id=str(task_data["id"]),
            instruction=str(task_data["instruction"]),
            target=tuple(int(value) for value in task_data["target"]),
            max_steps=int(task_data["max_steps"]),
        )

    async def run(
        self, sampling_params: dict[str, Any], priority: int = 0, **kwargs
    ) -> AgentLoopOutput:
        task = self._task_from_kwargs(kwargs)
        env = TextNavigationEnv(task)
        actions: list[str] = []
        raw_outputs: list[str] = []
        messages = list(kwargs["raw_prompt"])
        initial_prompt_ids = await self.apply_chat_template(messages)
        running_ids = list(initial_prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] | None = []
        request_id = uuid4().hex
        total_generate_seconds = 0.0
        total_preempted = 0
        extra_fields: dict[str, Any] = {}
        terminal_reason = "running"

        while not env.terminated:
            remaining_response_tokens = self.response_length - len(response_mask)
            if remaining_response_tokens <= 0:
                terminal_reason = "response_budget_exhausted"
                break
            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = min(
                self.action_max_new_tokens, remaining_response_tokens
            )

            started = time.monotonic()
            token_output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=running_ids,
                sampling_params=turn_sampling_params,
                priority=int(priority),
            )
            total_generate_seconds += time.monotonic() - started
            total_preempted += int(token_output.num_preempted or 0)
            if not extra_fields:
                extra_fields.update(token_output.extra_fields)
            elif token_output.extra_fields.get("max_global_steps") is not None:
                extra_fields["max_global_steps"] = token_output.extra_fields[
                    "max_global_steps"
                ]

            generated_ids = list(token_output.token_ids)
            available = self.response_length - len(response_mask)
            generated_ids = generated_ids[:available]
            raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            raw_outputs.append(raw_output)
            running_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            if response_logprobs is not None:
                if token_output.log_probs is None:
                    response_logprobs = None
                else:
                    response_logprobs.extend(
                        token_output.log_probs[: len(generated_ids)]
                    )

            action = parse_action(raw_output)
            if action is None:
                env.step("INVALID")
            else:
                actions.append(action)
                env.step(action)
            terminal_reason = env.reason
            if env.terminated:
                break

            feedback_ids = await self.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": build_agent_feedback(env, actions),
                    }
                ],
                remove_system_prompt=True,
            )
            feedback_ids = list(self.turn_separator) + list(feedback_ids)
            available = self.response_length - len(response_mask)
            if len(feedback_ids) >= available:
                terminal_reason = "response_budget_exhausted"
                break
            running_ids.extend(feedback_ids)
            response_mask.extend([0] * len(feedback_ids))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(feedback_ids))

        response_ids = running_ids[len(initial_prompt_ids) :]
        result = env.result()
        extra_fields.update(
            {
                "task_id": task.task_id,
                "actions": actions,
                "raw_outputs": raw_outputs,
                "final_position": list(result.position),
                "steps_used": result.steps_used,
                "success": result.success,
                "terminal_reason": terminal_reason,
            }
        )
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=1.0 if result.success else 0.0,
            num_turns=2 + len(raw_outputs) + max(0, len(raw_outputs) - 1),
            metrics={
                "generate_sequences": total_generate_seconds,
                "num_preempted": total_preempted,
            },
            extra_fields=extra_fields,
        )
