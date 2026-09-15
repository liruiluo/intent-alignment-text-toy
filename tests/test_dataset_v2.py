import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from intent_alignment_toy.dataset import (
    load_records,
    load_tasks,
    validate_split_pair,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "intent_alignment_train_400.jsonl"
TEST = ROOT / "data" / "intent_alignment_test_100.jsonl"
MANIFEST = ROOT / "data" / "dataset_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_400_100_split_contract():
    train_records = load_records(TRAIN)
    test_records = load_records(TEST)
    summary = validate_split_pair(train_records, test_records)

    assert summary["train_count"] == 400
    assert summary["test_count"] == 100
    assert set(summary["train_template_families"]).isdisjoint(
        summary["test_template_families"]
    )
    assert set(summary["train_cell_counts"].values()) == {20}
    assert set(summary["test_cell_counts"].values()) == {5}

    train_tasks = load_tasks(TRAIN, expected_count=400)
    test_tasks = load_tasks(TEST, expected_count=100)
    assert Counter(
        abs(x) + abs(y) for x, y in (task.target for task in train_tasks)
    ) == {
        1: 80,
        2: 80,
        3: 80,
        4: 80,
        5: 80,
    }
    assert Counter(
        abs(x) + abs(y) for x, y in (task.target for task in test_tasks)
    ) == {
        1: 20,
        2: 20,
        3: 20,
        4: 20,
        5: 20,
    }


def test_manifest_hashes_match_frozen_files():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["files"][TRAIN.name]["sha256"] == sha256(TRAIN)
    assert manifest["files"][TEST.name]["sha256"] == sha256(TEST)


def test_generator_is_byte_deterministic(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_dataset.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert (tmp_path / TRAIN.name).read_bytes() == TRAIN.read_bytes()
    assert (tmp_path / TEST.name).read_bytes() == TEST.read_bytes()
    regenerated = json.loads((tmp_path / MANIFEST.name).read_text(encoding="utf-8"))
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert regenerated == frozen
