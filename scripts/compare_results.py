#!/usr/bin/env python3
"""Create a strict paired before/after report for frozen test100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN_TEST = ROOT / "data" / "intent_alignment_test_100.jsonl"
EXPECTED_PROTOCOL_NAME = "intent-alignment-text-toy-v2-agent-loop"
EXPECTED_PROMPT_CONTRACT = "v2-agent-loop"
EXPECTED_DATASET_SHA256 = (
    "b2f7560c4466949136bec09bd2e8532a27def45696c5eeed3b4fcc99de2c1675"
)
EXPECTED_DECODING = {
    "do_sample": False,
    "max_new_tokens": 8,
    "thinking": False,
}
EXPECTED_BATCH_SIZE = 100
EXPECTED_MODEL_CONFIG_SHA256 = (
    "496d39ff3964cea8009c004e5d89fbd7801526992f699ff3f2934b12cb7e8892"
)
EXPECTED_TOKENIZER_FINGERPRINT = {
    "class": "TokenizersBackend",
    "vocab_size": 248077,
    "vocab_sha256": "5d326ac57a518b9102ed33be576eea8e2ab54c2f328ac6d424ffef4c2c53ef0a",
    "chat_template_sha256": "12de9034d5269e8b1fe6012b3538374fcb3941bf93c95fc81b0cf5acf3f08ce4",
    "bos_token_id": None,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "unk_token_id": None,
}
EXPECTED_RUNTIME = {
    "python": "3.12.12",
    "torch": "2.11.0+cu130",
    "transformers": "5.5.3",
    "device": "NVIDIA B300 SXM6 AC",
}
RUNTIME_IDENTITY_FIELDS = tuple(EXPECTED_RUNTIME)
TASK_IDENTITY_FIELDS = ("instruction", "target", "max_steps")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_tasks() -> dict[str, dict[str, Any]]:
    actual_hash = sha256_file(FROZEN_TEST)
    if actual_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "local frozen test100 drifted: "
            f"expected {EXPECTED_DATASET_SHA256}, got {actual_hash}"
        )
    rows = [
        json.loads(line)
        for line in FROZEN_TEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise ValueError(f"frozen test split must contain 100 tasks, got {len(rows)}")
    return {row["id"]: row for row in rows}


def weight_digests(payload: dict[str, Any], label: str) -> dict[str, str]:
    files = payload.get("model", {}).get("weights", {}).get("files", [])
    if not files:
        raise ValueError(f"{label}: weight manifest is empty")
    digests = {item["name"]: item["sha256"] for item in files}
    if len(digests) != len(files):
        raise ValueError(f"{label}: duplicate weight filenames")
    return digests


def validate_payload(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    protocol = payload.get("protocol", {})
    expected_tasks = frozen_tasks()
    checks = {
        "protocol name": (protocol.get("name"), EXPECTED_PROTOCOL_NAME),
        "dataset name": (protocol.get("dataset"), FROZEN_TEST.name),
        "dataset hash": (
            protocol.get("dataset_sha256"),
            EXPECTED_DATASET_SHA256,
        ),
        "prompt contract": (
            protocol.get("prompt_contract"),
            EXPECTED_PROMPT_CONTRACT,
        ),
        "decoding": (protocol.get("decoding"), EXPECTED_DECODING),
        "evaluation batch size": (
            protocol.get("evaluation_batch_size"),
            EXPECTED_BATCH_SIZE,
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"{label}: {field} drifted: expected {expected!r}, got {actual!r}"
            )

    rows = payload.get("tasks", [])
    if len(rows) != len(expected_tasks):
        raise ValueError(f"{label}: expected 100 task results, got {len(rows)}")
    tasks = {row["id"]: row for row in rows}
    if len(tasks) != len(rows) or tasks.keys() != expected_tasks.keys():
        raise ValueError(f"{label}: task identities differ from frozen test100")
    for task_id, row in tasks.items():
        expected = expected_tasks[task_id]
        for field in TASK_IDENTITY_FIELDS:
            if row.get(field) != expected.get(field):
                raise ValueError(f"{label}: {task_id} {field} drifted")
        if not isinstance(row.get("success"), bool):
            raise TypeError(f"{label}: {task_id} success is not boolean")

    summary = payload.get("summary", {})
    successes = sum(row["success"] for row in rows)
    if summary.get("total") != 100 or summary.get("successes") != successes:
        raise ValueError(f"{label}: summary count is inconsistent with task rows")
    if summary.get("success_rate") != successes / 100:
        raise ValueError(f"{label}: summary success rate is inconsistent")

    model = payload.get("model", {})
    if model.get("config_sha256") != EXPECTED_MODEL_CONFIG_SHA256:
        raise ValueError(f"{label}: model config is not the frozen Qwen3.5-4B config")
    if model.get("tokenizer_fingerprint") != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ValueError(f"{label}: tokenizer fingerprint drifted")
    runtime = payload.get("runtime", {})
    for field, expected in EXPECTED_RUNTIME.items():
        if runtime.get(field) != expected:
            raise ValueError(
                f"{label}: runtime field {field!r} drifted: "
                f"expected {expected!r}, got {runtime.get(field)!r}"
            )
    weight_digests(payload, label)
    return tasks


def build_report(
    before: dict[str, Any],
    after: dict[str, Any],
    before_path: Path,
    after_path: Path,
) -> dict[str, Any]:
    before_tasks = validate_payload(before, "before")
    after_tasks = validate_payload(after, "after")

    if before["protocol"] != after["protocol"]:
        raise ValueError("before/after protocol objects differ")
    if (
        before["model"]["tokenizer_fingerprint"]
        != after["model"]["tokenizer_fingerprint"]
    ):
        raise ValueError("before/after tokenizer fingerprints differ")
    if before["model"]["config_sha256"] != after["model"]["config_sha256"]:
        raise ValueError("before/after model configs differ")
    for field in RUNTIME_IDENTITY_FIELDS:
        if before["runtime"][field] != after["runtime"][field]:
            raise ValueError(f"before/after runtime field {field!r} differs")

    improved: list[str] = []
    regressed: list[str] = []
    unchanged_success: list[str] = []
    unchanged_failure: list[str] = []
    for task_id in before_tasks:
        for field in TASK_IDENTITY_FIELDS:
            if before_tasks[task_id][field] != after_tasks[task_id][field]:
                raise ValueError(f"before/after {task_id} {field} differs")
        pair = (before_tasks[task_id]["success"], after_tasks[task_id]["success"])
        if pair == (False, True):
            improved.append(task_id)
        elif pair == (True, False):
            regressed.append(task_id)
        elif pair == (True, True):
            unchanged_success.append(task_id)
        else:
            unchanged_failure.append(task_id)

    before_weights = weight_digests(before, "before")
    after_weights = weight_digests(after, "after")
    if before_weights.keys() != after_weights.keys():
        raise ValueError("before/after weight file sets differ")
    changed_weights = sorted(
        name for name in before_weights if before_weights[name] != after_weights[name]
    )
    if not changed_weights:
        raise ValueError("no changed weight file was observed after training")

    return {
        "before_result": str(before_path.resolve()),
        "after_result": str(after_path.resolve()),
        "before_result_sha256": sha256_file(before_path),
        "after_result_sha256": sha256_file(after_path),
        "dataset_sha256": before["protocol"]["dataset_sha256"],
        "prompt_contract": before["protocol"]["prompt_contract"],
        "decoding": before["protocol"]["decoding"],
        "evaluation_batch_size": before["protocol"]["evaluation_batch_size"],
        "before": before["summary"],
        "after": after["summary"],
        "success_rate_delta": (
            after["summary"]["success_rate"] - before["summary"]["success_rate"]
        ),
        "paired": {
            "improved": improved,
            "regressed": regressed,
            "unchanged_success": unchanged_success,
            "unchanged_failure": unchanged_failure,
        },
        "evaluation_identity": {
            "tokenizer_fingerprint": before["model"]["tokenizer_fingerprint"],
            "model_config_sha256": before["model"]["config_sha256"],
            "runtime": {
                field: before["runtime"][field] for field in RUNTIME_IDENTITY_FIELDS
            },
        },
        "parameter_evidence": {
            "weight_files": sorted(before_weights),
            "changed_weight_files": changed_weights,
            "weights_changed": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    report = build_report(load(before_path), load(after_path), before_path, after_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
