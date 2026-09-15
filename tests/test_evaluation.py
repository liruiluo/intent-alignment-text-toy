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
        for task, env, _history in requests:
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
