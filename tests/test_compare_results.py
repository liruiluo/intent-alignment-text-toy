import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_results", ROOT / "scripts" / "compare_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def payload(weight_digest: str) -> dict:
    frozen = [
        json.loads(line)
        for line in MODULE.FROZEN_TEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = [
        {
            "id": row["id"],
            "instruction": row["instruction"],
            "target": row["target"],
            "max_steps": row["max_steps"],
            "success": True,
        }
        for row in frozen
    ]
    return {
        "protocol": {
            "name": MODULE.EXPECTED_PROTOCOL_NAME,
            "dataset": MODULE.FROZEN_TEST.name,
            "dataset_sha256": MODULE.EXPECTED_DATASET_SHA256,
            "prompt_contract": MODULE.EXPECTED_PROMPT_CONTRACT,
            "decoding": copy.deepcopy(MODULE.EXPECTED_DECODING),
            "evaluation_batch_size": MODULE.EXPECTED_BATCH_SIZE,
        },
        "model": {
            "config_sha256": MODULE.EXPECTED_MODEL_CONFIG_SHA256,
            "tokenizer_fingerprint": copy.deepcopy(
                MODULE.EXPECTED_TOKENIZER_FINGERPRINT
            ),
            "weights": {
                "files": [{"name": "model.safetensors", "sha256": weight_digest}]
            },
        },
        "runtime": copy.deepcopy(MODULE.EXPECTED_RUNTIME),
        "summary": {"total": 100, "successes": 100, "success_rate": 1.0},
        "tasks": tasks,
    }


def build(before: dict, after: dict, tmp_path: Path) -> dict:
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")
    return MODULE.build_report(before, after, before_path, after_path)


def test_strict_comparison_accepts_paired_protocol_and_changed_weights(tmp_path):
    report = build(payload("a" * 64), payload("b" * 64), tmp_path)
    assert report["parameter_evidence"]["weights_changed"]
    assert len(report["paired"]["unchanged_success"]) == 100


def test_comparison_rejects_decoding_drift(tmp_path):
    before = payload("a" * 64)
    after = payload("b" * 64)
    after["protocol"]["decoding"]["max_new_tokens"] = 1
    with pytest.raises(ValueError, match="decoding drifted"):
        build(before, after, tmp_path)


def test_comparison_rejects_task_identity_drift(tmp_path):
    before = payload("a" * 64)
    after = payload("b" * 64)
    after["tasks"][0]["instruction"] += " changed"
    with pytest.raises(ValueError, match="instruction drifted"):
        build(before, after, tmp_path)


def test_comparison_rejects_empty_or_unchanged_weight_evidence(tmp_path):
    before = payload("a" * 64)
    after = payload("b" * 64)
    after["model"]["weights"]["files"] = []
    with pytest.raises(ValueError, match="weight manifest is empty"):
        build(before, after, tmp_path)

    with pytest.raises(ValueError, match="no changed weight file"):
        build(before, payload("a" * 64), tmp_path)


def test_comparison_rejects_tokenizer_or_runtime_drift(tmp_path):
    before = payload("a" * 64)
    after = payload("b" * 64)
    after["model"]["tokenizer_fingerprint"]["vocab_size"] += 1
    with pytest.raises(ValueError, match="tokenizer fingerprint drifted"):
        build(before, after, tmp_path)

    after = payload("b" * 64)
    after["runtime"]["device"] = "NVIDIA A100-SXM4-80GB"
    with pytest.raises(ValueError, match="runtime field 'device' drifted"):
        build(before, after, tmp_path)


def test_comparator_pins_frozen_dataset_hash():
    assert MODULE.EXPECTED_DATASET_SHA256 == (
        "b2f7560c4466949136bec09bd2e8532a27def45696c5eeed3b4fcc99de2c1675"
    )
    assert MODULE.sha256_file(MODULE.FROZEN_TEST) == MODULE.EXPECTED_DATASET_SHA256


def test_comparator_rejects_mutated_local_test_file(tmp_path, monkeypatch):
    drifted = tmp_path / MODULE.FROZEN_TEST.name
    drifted.write_bytes(MODULE.FROZEN_TEST.read_bytes() + b"\n")
    monkeypatch.setattr(MODULE, "FROZEN_TEST", drifted)

    with pytest.raises(ValueError, match="local frozen test100 drifted"):
        MODULE.frozen_tasks()
