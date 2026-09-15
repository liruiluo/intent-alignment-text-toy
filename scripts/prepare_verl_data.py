#!/usr/bin/env python3
"""Convert frozen JSONL splits to the parquet schema consumed by veRL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from intent_alignment_toy.dataset import load_records, validate_split_pair
from intent_alignment_toy.environment import Task
from intent_alignment_toy.evaluation import SYSTEM_PROMPT, build_initial_agent_prompt

AGENT_NAME = "intent_alignment_agent"
DATA_SOURCE = "intent_alignment_text_toy_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_verl_row(record: dict[str, Any], index: int) -> dict[str, Any]:
    task = Task(
        task_id=record["id"],
        instruction=record["instruction"],
        target=tuple(record["target"]),
        max_steps=record["max_steps"],
    )
    return {
        "data_source": DATA_SOURCE,
        "agent_name": AGENT_NAME,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_agent_prompt(task)},
        ],
        "ability": "intent_alignment",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {"target": record["target"], "max_steps": record["max_steps"]},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
        "extra_info": {
            "index": index,
            "split": record["split"],
            "task": {
                "id": record["id"],
                "instruction": record["instruction"],
                "target": record["target"],
                "max_steps": record["max_steps"],
                "direction": record["direction"],
                "distance": record["distance"],
                "template_family": record["template_family"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-jsonl", default="data/intent_alignment_train_400.jsonl"
    )
    parser.add_argument("--test-jsonl", default="data/intent_alignment_test_100.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from datasets import Dataset

    train_jsonl = Path(args.train_jsonl).resolve()
    test_jsonl = Path(args.test_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(train_jsonl)
    test_records = load_records(test_jsonl)
    split_validation = validate_split_pair(train_records, test_records)

    outputs = {}
    for split, source_path, records in (
        ("train", train_jsonl, train_records),
        ("test", test_jsonl, test_records),
    ):
        rows = [build_verl_row(record, index) for index, record in enumerate(records)]
        output_path = output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(str(output_path))
        readback = Dataset.from_parquet(str(output_path))
        if len(readback) != len(rows):
            raise RuntimeError(f"{split}: parquet readback count drifted")
        if readback[0]["extra_info"]["task"]["id"] != records[0]["id"]:
            raise RuntimeError(f"{split}: first-row identity drifted")
        if readback[-1]["extra_info"]["task"]["id"] != records[-1]["id"]:
            raise RuntimeError(f"{split}: last-row identity drifted")
        outputs[split] = {
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "parquet": str(output_path),
            "parquet_sha256": sha256_file(output_path),
            "count": len(readback),
            "columns": readback.column_names,
            "first_task_id": records[0]["id"],
            "last_task_id": records[-1]["id"],
        }

    manifest = {
        "protocol": "intent-alignment-text-toy-v2-verl",
        "agent_name": AGENT_NAME,
        "data_source": DATA_SOURCE,
        "split_validation": split_validation,
        "outputs": outputs,
    }
    manifest_path = output_dir / "verl_data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
