from collections import Counter

from intent_alignment_toy.dataset import load_tasks


def test_frozen_dataset_contract():
    tasks = load_tasks("data/intent_alignment_20.jsonl")
    assert len(tasks) == 20
    distances = [abs(task.target[0]) + abs(task.target[1]) for task in tasks]
    assert Counter(distances) == Counter({length: 4 for length in range(1, 6)})
    assert max(task.max_steps for task in tasks) == 5
