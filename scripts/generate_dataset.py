#!/usr/bin/env python3
"""Generate the deterministic 400-train/100-test intent-alignment dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from intent_alignment_toy.dataset import validate_split_pair

DIRECTIONS = {
    "RIGHT": {"word": "右", "place": "右侧", "target": lambda d: [d, 0]},
    "LEFT": {"word": "左", "place": "左侧", "target": lambda d: [-d, 0]},
    "UP": {"word": "上", "place": "上方", "target": lambda d: [0, d]},
    "DOWN": {"word": "下", "place": "下方", "target": lambda d: [0, -d]},
}

TRAIN_TEMPLATES = (
    "请在{budget}步以内向{word}走{distance}米。",
    "最多走{budget}步，向{word}移动{distance}米。",
    "动作上限为{budget}步，请前往{place}{distance}米处。",
    "用不超过{budget}步完成向{word}{distance}米的位移。",
    "请在{budget}次移动内到达{place}{distance}米的位置。",
    "从原点开始，向{word}前进{distance}米，最多允许{budget}步。",
    "限{budget}个动作，请移动到{place}{distance}米处。",
    "在至多{budget}步内，沿{word}方向移动{distance}米。",
    "你有{budget}次移动机会，请向{word}走{distance}米。",
    "请把位置移到{place}{distance}米处，动作不得超过{budget}次。",
    "移动预算是{budget}步，目标是向{word}{distance}米。",
    "不要超过{budget}步，完成向{word}移动{distance}米。",
    "请用最多{budget}次动作抵达{place}{distance}米处。",
    "在{budget}步的预算内，沿{word}方向前进{distance}米。",
    "目标位于{place}{distance}米，要求{budget}步内到达。",
    "可执行至多{budget}次移动，请向{word}走{distance}米。",
    "请控制在{budget}步以内，移动到{place}{distance}米。",
    "从当前位置向{word}移动{distance}米，步数上限是{budget}。",
    "任务是向{word}走{distance}米，你最多能走{budget}步。",
    "请在不多于{budget}次动作内到达{place}{distance}米处。",
)

TEST_TEMPLATES = (
    "终点在你的{place}{distance}米处，最多行动{budget}次。",
    "请把当前位置沿{word}方向改变{distance}米，动作次数不得超过{budget}。",
    "由原点出发，限{budget}步，抵达{place}{distance}米的位置。",
    "你需要在{budget}次移动内完成{distance}米的向{word}位移。",
    "可走{budget}步，请到达距离起点{place}{distance}米的终点。",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def budget_for(direction_index: int, distance: int, family_index: int) -> int:
    """Choose a deterministic valid budget while covering slack 0..(5-distance)."""
    slack_options = 6 - distance
    return distance + ((direction_index + family_index) % slack_options)


def generate_split(split: str, templates: tuple[str, ...]) -> list[dict]:
    records: list[dict] = []
    for direction_index, (direction, fields) in enumerate(DIRECTIONS.items()):
        for distance in range(1, 6):
            for family_index, template in enumerate(templates, start=1):
                budget = budget_for(direction_index, distance, family_index - 1)
                family = f"{split}-f{family_index:02d}"
                records.append(
                    {
                        "id": f"{split}-{direction.lower()}-{distance}-f{family_index:02d}",
                        "split": split,
                        "template_family": family,
                        "instruction": template.format(
                            budget=budget,
                            word=fields["word"],
                            place=fields["place"],
                            distance=distance,
                        ),
                        "direction": direction,
                        "distance": distance,
                        "target": fields["target"](distance),
                        "max_steps": budget,
                    }
                )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_path = output_dir / "intent_alignment_train_400.jsonl"
    test_path = output_dir / "intent_alignment_test_100.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"

    train_records = generate_split("train", TRAIN_TEMPLATES)
    test_records = generate_split("test", TEST_TEMPLATES)
    validation = validate_split_pair(train_records, test_records)
    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)

    manifest = {
        "protocol": "intent-alignment-text-toy-v2",
        "generator": str(Path(__file__).name),
        "generator_sha256": sha256_file(Path(__file__)),
        "generation_is_deterministic": True,
        "split_rule": "20 train and 5 test lexical template families; no family overlap",
        "semantic_grid": "4 cardinal directions x distances 1..5",
        "action_contract": "UP/DOWN/LEFT/RIGHT; exactly 1 metre per action; max_steps <= 5",
        "files": {
            train_path.name: {
                "sha256": sha256_file(train_path),
                "count": len(train_records),
            },
            test_path.name: {
                "sha256": sha256_file(test_path),
                "count": len(test_records),
            },
        },
        "validation": validation,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
