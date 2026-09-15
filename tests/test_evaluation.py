from intent_alignment_toy.dataset import load_tasks
from intent_alignment_toy.evaluation import evaluate_with_policy, parse_action


def test_action_parser_is_strict():
    assert parse_action("RIGHT") == "RIGHT"
    assert parse_action("  UP\n") == "UP"
    assert parse_action("向右") is None
    assert parse_action("RIGHT.") is None
    assert parse_action("```RIGHT```") is None


def test_oracle_scores_twenty_of_twenty():
    tasks = load_tasks("data/intent_alignment_20.jsonl")

    def oracle(requests):
        actions = []
        for task, env, _history, _raw_outputs in requests:
            x, y = env.position
            target_x, target_y = task.target
            if x < target_x:
                actions.append("RIGHT")
            elif x > target_x:
                actions.append("LEFT")
            elif y < target_y:
                actions.append("UP")
            else:
                actions.append("DOWN")
        return actions

    result = evaluate_with_policy(tasks, oracle)
    assert result["summary"]["successes"] == 20
    assert result["summary"]["meets_at_least_95_percent"]
    assert result["summary"]["meets_strictly_above_95_percent"]


def test_prompt_declares_coordinate_semantics():
    tasks = load_tasks("data/intent_alignment_20.jsonl")
    from intent_alignment_toy.environment import TextNavigationEnv
    from intent_alignment_toy.evaluation import build_policy_prompt

    prompt = build_policy_prompt(tasks[0], TextNavigationEnv(tasks[0]), [])
    assert "DOWN=(0,-1)" in prompt
    assert "LEFT=(-1,0)" in prompt
