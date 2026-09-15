#!/usr/bin/env python3
"""Verify every shipped artifact for the frozen one-step veRL result."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "results" / "qwen3.5-4b-verl-one-step-v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRIVATE_MARKERS = (
    "/home/",
    "/Users/",
    "/dev/shm",
    "6.235.",
    "luolirui-1-",
    "master-0",
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparator():
    path = ROOT / "scripts" / "compare_results.py"
    spec = importlib.util.spec_from_file_location("strict_result_comparator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load strict result comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(result_dir: Path) -> dict[str, Any]:
    manifest = load_json(result_dir / "artifact-manifest.json")
    if manifest.get("schema") != "intent_alignment_public_result_manifest_v2":
        raise ValueError("unsupported publication manifest schema")
    published = manifest.get("published_artifact_sha256")
    if not isinstance(published, dict) or not published:
        raise ValueError("published artifact manifest is empty")
    for name, expected in published.items():
        path = result_dir / name
        if not path.is_file():
            raise ValueError(f"published artifact is missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"published artifact hash drifted for {name}: {actual} != {expected}"
            )

    sources = manifest.get("source_artifact_sha256")
    if not isinstance(sources, dict) or any(
        not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
        for value in sources.values()
    ):
        raise ValueError("source artifact hash manifest is invalid")

    for path in result_dir.iterdir():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_MARKERS:
            if marker in text:
                raise ValueError(f"{path.name} contains private marker {marker!r}")

    paths = {
        "before": result_dir / "before-test100.json",
        "after": result_dir / "after-test100.json",
        "parameter": result_dir / "parameter-evidence.json",
        "training": result_dir / "training-evidence.json",
        "rollout": result_dir / "rollouts-step1.jsonl",
        "training_log": result_dir / "training-log.redacted.txt",
        "checkpoint": result_dir / "checkpoint-inventory.json",
        "run_contract": result_dir / "run-contract.redacted.json",
        "resolved_config": result_dir / "resolved-config.redacted.yaml",
        "launcher": result_dir / "launch.redacted.sh",
    }
    comparator = load_comparator()
    rebuilt = comparator.build_report(
        load_json(paths["before"]),
        load_json(paths["after"]),
        load_json(paths["parameter"]),
        load_json(paths["training"]),
        paths["before"],
        paths["after"],
        paths["parameter"],
        paths["training"],
        paths["rollout"],
        paths["training_log"],
        paths["checkpoint"],
        paths["run_contract"],
        paths["resolved_config"],
        paths["launcher"],
    )
    paired = load_json(result_dir / "paired-comparison.json")
    source_paired_hash = paired.pop("source_artifact_sha256", None)
    if source_paired_hash != sources.get("paired-comparison.json"):
        raise ValueError("paired comparison source binding drifted")
    if paired != rebuilt:
        raise ValueError("paired comparison is not a deterministic rebuild")

    training = load_json(paths["training"])
    parameter = load_json(paths["parameter"])
    complete = load_json(result_dir / "training-complete.json")
    status = load_json(result_dir / "result-status.json")
    if status.get("status") != "complete":
        raise ValueError("result status is not complete")
    if status.get("global_step") != 1 or status.get("optimizer_steps_actual") != 1:
        raise ValueError("result status optimizer count drifted")
    if complete.get("global_step") != 1 or complete.get("optimizer_updates_actual") != 1:
        raise ValueError("training completion optimizer count drifted")
    if len(
        {
            training["checkpoint_inventory_sha256"],
            parameter["checkpoint_inventory_sha256"],
            complete["checkpoint_inventory_sha256"],
        }
    ) != 1:
        raise ValueError("checkpoint inventory cross-binding drifted")
    expected_status_artifacts = {
        name: digest
        for name, digest in published.items()
        if name != "result-status.json"
    }
    if status.get("artifacts") != expected_status_artifacts:
        raise ValueError("result status artifact map drifted")

    return {
        "status": "PASS",
        "published_artifacts": len(published),
        "source_artifacts": len(sources),
        "before_successes": rebuilt["before"]["successes"],
        "after_successes": rebuilt["after"]["successes"],
        "optimizer_updates": training["optimizer_updates_actual"],
        "trajectories": training["trajectory_count"],
        "mixed_reward_groups": training["mixed_reward_group_count"],
        "changed_probes": parameter["changed_probe_count"],
        "effective_bfloat16_changed_values": parameter[
            "effective_bfloat16_changed_value_count"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    print(json.dumps(verify(args.result_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
