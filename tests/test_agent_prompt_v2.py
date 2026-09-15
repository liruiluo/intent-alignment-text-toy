from intent_alignment_toy.environment import Task, TextNavigationEnv
from intent_alignment_toy.evaluation import (
    build_agent_feedback,
    build_agent_messages,
    build_initial_agent_prompt,
)


def test_v2_conversation_replays_agent_loop_without_hiding_future_state():
    task = Task("right-3", "在5步以内向右走3米。", (3, 0), 5)
    messages = build_agent_messages(task, ["RIGHT", "RIGHT"])

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "当前位置：(0, 0)" in messages[1]["content"]
    assert "当前位置：(1, 0)" in messages[3]["content"]
    assert "当前位置：(2, 0)" in messages[5]["content"]
    assert "(3, 0)" not in messages[5]["content"]


def test_initial_and_feedback_prompts_keep_atomic_action_contract():
    task = Task("left-2", "向左走2米，最多3步。", (-2, 0), 3)
    initial = build_initial_agent_prompt(task)
    env = TextNavigationEnv(task)
    env.step("LEFT")
    feedback = build_agent_feedback(env, ["LEFT"])

    assert "每次动作恰好移动 1 米" in initial
    assert "每一轮只能选择一个动作" in initial
    assert "只输出下一步的一个动作标签" in feedback
    assert "剩余动作数：2" in feedback


def test_v2_conversation_preserves_exact_valid_assistant_text():
    task = Task("right-2", "向右走2米。", (2, 0), 2)
    messages = build_agent_messages(
        task,
        ["RIGHT"],
        raw_outputs=["  RIGHT\n"],
    )

    assert messages[2] == {"role": "assistant", "content": "  RIGHT\n"}
    assert "当前位置：(1, 0)" in messages[3]["content"]
