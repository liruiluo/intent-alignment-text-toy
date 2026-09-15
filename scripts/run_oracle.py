#!/usr/bin/env python3
"""Run a deterministic oracle to verify the environment and evaluation plumbing."""

import argparse
import json
from pathlib import Path

from intent_alignment_toy.dataset import load_tasks
from intent_alignment_toy.evaluation import evaluate_with_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/intent_alignment_20.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()
    tasks = load_tasks(args.dataset)

    def oracle(requests):
        outputs = []
        for task, env, _actions, _raw_outputs in requests:
            x, y = env.position
            target_x, target_y = task.target
            if x < target_x:
                outputs.append("RIGHT")
            elif x > target_x:
                outputs.append("LEFT")
            elif y < target_y:
                outputs.append("UP")
            else:
                outputs.append("DOWN")
        return outputs

    result = evaluate_with_policy(tasks, oracle)
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
