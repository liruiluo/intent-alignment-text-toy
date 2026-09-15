#!/usr/bin/env python3
"""Build a privacy-safe, independently checkable formal-result package.

The script never mutates the formal run. It copies the complete rollout ledger,
redacts machine-local strings from the training log, and binds those artifacts
plus the checkpoint inventory to the published JSON evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
IPV4_RE = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])")
FORBIDDEN_PUBLIC_MARKERS = (
    "/home/",
    "/Users/",
    "/dev/shm",
    "luolirui-1-",
    "master-0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ensure_private_markers_absent(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for marker in FORBIDDEN_PUBLIC_MARKERS:
        if marker in text:
            raise ValueError(f"{path.name} contains private marker {marker!r}")
    for match in IPV4_RE.finditer(text):
        octets = [int(part) for part in match.group(0).split(".")]
        if all(0 <= value <= 255 for value in octets) and match.group(0) != "127.0.0.1":
            raise ValueError(f"{path.name} contains IPv4 address {match.group(0)!r}")


def redact_machine_text(text: str, contract: dict[str, Any], run_dir: Path) -> str:
    replacements = {
        str(run_dir): "/REDACTED/RUN_DIR",
        str(contract["source"]["path"]): "/REDACTED/SOURCE_ROOT",
        str(contract["data"]["train_parquet"]): "/REDACTED/TRAIN_PARQUET",
        str(contract["data"]["test_parquet"]): "/REDACTED/TEST_PARQUET",
        str(contract["model"]["path"]): "/REDACTED/BASE_MODEL_ROOT",
        str(contract["runtime"]["python"]): "/REDACTED/PYTHON_BIN",
        str(contract["runtime"]["runtime_root"]): "/REDACTED/RUNTIME_ROOT",
        str(contract["runtime"]["cuda_home"]): "/REDACTED/CUDA_HOME",
        str(contract["runtime"]["transfer_queue_overlay"]): (
            "/REDACTED/TRANSFER_QUEUE_OVERLAY"
        ),
        str(contract["hardware"]["platform_task"]): "REDACTED_PLATFORM_TASK",
        str(contract["hardware"]["pod"]): "REDACTED_POD_NAME",
        str(contract["hardware"]["pod_ip"]): "REDACTED_POD_IP",
    }
    # Replace longest strings first so nested paths do not leak a suffix.
    redacted = ANSI_RE.sub("", text)
    for source, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        redacted = redacted.replace(source, replacement)
    redacted = re.sub(
        r"/home/ai-jingyan-train/luolirui\.1/post-train/"
        r"intent-alignment-text-toy-workspace/upstream/verl-v0\.9\.0",
        "/REDACTED/VERL_ROOT",
        redacted,
    )
    redacted = re.sub(r"/dev/shm/[^\s'\"]+", "/REDACTED/POD_SCRATCH", redacted)
    redacted = re.sub(r"/home/[^\s'\"]+", "/REDACTED/PRIVATE_PATH", redacted)
    redacted = re.sub(r"/Users/[^\s'\"]+", "/REDACTED/PRIVATE_PATH", redacted)
    redacted = redacted.replace("6.235.124.254", "REDACTED_POD_IP")
    had_final_newline = redacted.endswith("\n")
    redacted = "\n".join(line.rstrip() for line in redacted.splitlines())
    if had_final_newline:
        redacted += "\n"
    return redacted


def public_evaluation_payload(source_path: Path, label: str) -> dict[str, Any]:
    """Redact a raw evaluation artifact without relying on an older public copy."""

    if label not in {"before", "after"}:
        raise ValueError(f"unsupported evaluation label: {label!r}")
    payload = load_json(source_path)
    payload["model"]["path"] = (
        "Qwen3.5-4B frozen base checkpoint"
        if label == "before"
        else "global_step_1/actor/huggingface"
    )
    payload["runtime"]["hostname"] = "single-B300-runner"
    payload["source_artifact_sha256"] = sha256_file(source_path)
    return payload


def parse_checkpoint_inventory(
    size_file: Path, digest_file: Path, run_id: str
) -> dict[str, Any]:
    sizes: dict[str, int] = {}
    for line in size_file.read_text(encoding="utf-8").splitlines():
        path, size = line.split("\t", 1)
        sizes[f"global_step_1/{path}"] = int(size)
    digests: dict[str, str] = {}
    for line in digest_file.read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        path = path.removeprefix("checkpoints/")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid checkpoint digest for {path!r}")
        digests[path] = digest
    if sizes.keys() != digests.keys():
        raise ValueError("checkpoint size and digest inventories differ")
    return {
        "schema": "intent_alignment_checkpoint_inventory_v1",
        "run_id": run_id,
        "global_step": 1,
        "files": [
            {"path": path, "bytes": sizes[path], "sha256": digests[path]}
            for path in sorted(sizes)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--public-evaluation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sizes", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_json(run_dir / "contract" / "run-contract.json")
    raw_evaluation_dir = run_dir / "evaluation"
    before = public_evaluation_payload(
        raw_evaluation_dir / "before-test100.json", "before"
    )
    after = public_evaluation_payload(
        raw_evaluation_dir / "after-test100.json", "after"
    )
    write_json(output_dir / "before-test100.json", before)
    write_json(output_dir / "after-test100.json", after)
    run_id = contract["run_id"]
    declared_run_dir = Path(contract["launcher"]["path"]).parent.parent

    redacted_sources = {
        "run_contract": (
            run_dir / "contract" / "run-contract.json",
            output_dir / "run-contract.redacted.json",
        ),
        "resolved_config": (
            run_dir / "contract" / "resolved-config.yaml",
            output_dir / "resolved-config.redacted.yaml",
        ),
        "launcher": (
            run_dir / "contract" / "launch.sh",
            output_dir / "launch.redacted.sh",
        ),
    }
    for source_path, public_path in redacted_sources.values():
        public_path.write_text(
            redact_machine_text(
                source_path.read_text(encoding="utf-8"), contract, declared_run_dir
            ),
            encoding="utf-8",
        )
        ensure_private_markers_absent(public_path)
        if public_path.suffix == ".json":
            load_json(public_path)

    rollout_source = run_dir / "rollouts" / "1.jsonl"
    rollout_public = output_dir / "rollouts-step1.jsonl"
    shutil.copyfile(rollout_source, rollout_public)
    ensure_private_markers_absent(rollout_public)

    log_source = run_dir / "logs" / "training.log"
    log_public = output_dir / "training-log.redacted.txt"
    log_public.write_text(
        redact_machine_text(
            log_source.read_text(encoding="utf-8"), contract, declared_run_dir
        ),
        encoding="utf-8",
    )
    ensure_private_markers_absent(log_public)

    inventory = parse_checkpoint_inventory(
        args.checkpoint_sizes, args.checkpoint_sha256, run_id
    )
    inventory_path = output_dir / "checkpoint-inventory.json"
    write_json(inventory_path, inventory)

    raw_training = load_json(run_dir / "training-evidence.json")
    raw_parameter = load_json(run_dir / "parameter-evidence.json")
    raw_complete = load_json(run_dir / "training-complete.json")

    training = dict(raw_training)
    training.update(
        {
            "schema": "intent_alignment_one_step_training_evidence_v2",
            "optimizer_updates_actual": 1,
            "source_commit": contract["source"]["commit"],
            "verl_commit": contract["runtime"]["verl_commit"],
            "train_jsonl_sha256": contract["data"]["train_jsonl_sha256"],
            "test_jsonl_sha256": contract["data"]["test_jsonl_sha256"],
            "train_parquet_sha256": contract["data"]["train_parquet_sha256"],
            "test_parquet_sha256": contract["data"]["test_parquet_sha256"],
            "source_run_contract_sha256": sha256_file(
                redacted_sources["run_contract"][0]
            ),
            "published_run_contract_sha256": sha256_file(
                redacted_sources["run_contract"][1]
            ),
            "run_contract_file": redacted_sources["run_contract"][1].name,
            "source_resolved_config_sha256": sha256_file(
                redacted_sources["resolved_config"][0]
            ),
            "published_resolved_config_sha256": sha256_file(
                redacted_sources["resolved_config"][1]
            ),
            "resolved_config_file": redacted_sources["resolved_config"][1].name,
            "source_launcher_sha256": sha256_file(
                redacted_sources["launcher"][0]
            ),
            "published_launcher_sha256": sha256_file(
                redacted_sources["launcher"][1]
            ),
            "launcher_file": redacted_sources["launcher"][1].name,
            "source_training_log_sha256": sha256_file(log_source),
            "published_training_log_sha256": sha256_file(log_public),
            "training_log_file": log_public.name,
            "source_rollout_sha256": sha256_file(rollout_source),
            "published_rollout_sha256": sha256_file(rollout_public),
            "rollout_file": rollout_public.name,
            "checkpoint_inventory_sha256": sha256_file(inventory_path),
            "checkpoint_inventory_file": inventory_path.name,
            "source_artifact_sha256": sha256_file(
                run_dir / "training-evidence.json"
            ),
        }
    )
    write_json(output_dir / "training-evidence.json", training)

    parameter = dict(raw_parameter)
    parameter.update(
        {
            "schema": "intent_alignment_semantic_parameter_evidence_v2",
            "base_path": "Qwen3.5-4B frozen base checkpoint",
            "after_path": "global_step_1/actor/huggingface",
            "evaluated_weight_files": {
                "before": sorted(
                    before["model"]["weights"]["files"], key=lambda row: row["name"]
                ),
                "after": sorted(
                    after["model"]["weights"]["files"], key=lambda row: row["name"]
                ),
            },
            "checkpoint_inventory_sha256": sha256_file(inventory_path),
            "source_artifact_sha256": sha256_file(
                run_dir / "parameter-evidence.json"
            ),
        }
    )
    write_json(output_dir / "parameter-evidence.json", parameter)

    complete = dict(raw_complete)
    complete.update(
        {
            "optimizer_updates_actual": 1,
            "checkpoint_inventory_file": inventory_path.name,
            "checkpoint_inventory_sha256": sha256_file(inventory_path),
            "source_artifact_sha256": sha256_file(
                run_dir / "training-complete.json"
            ),
        }
    )
    complete_path = output_dir / "training-complete.json"
    write_json(complete_path, complete)

    import importlib.util

    comparator_path = Path(__file__).with_name("compare_results.py")
    spec = importlib.util.spec_from_file_location(
        "intent_alignment_strict_comparator", comparator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load strict result comparator")
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)

    before_path = output_dir / "before-test100.json"
    after_path = output_dir / "after-test100.json"
    parameter_path = output_dir / "parameter-evidence.json"
    training_path = output_dir / "training-evidence.json"
    paired = comparator.build_report(
        before,
        after,
        parameter,
        training,
        before_path,
        after_path,
        parameter_path,
        training_path,
        rollout_public,
        log_public,
        inventory_path,
        redacted_sources["run_contract"][1],
        redacted_sources["resolved_config"][1],
        redacted_sources["launcher"][1],
    )
    paired["source_artifact_sha256"] = sha256_file(
        run_dir / "evaluation" / "paired-comparison.json"
    )
    paired_path = output_dir / "paired-comparison.json"
    write_json(paired_path, paired)

    raw_status_path = run_dir / "result-status.json"
    status = load_json(raw_status_path)
    status.pop("hardware_task", None)
    status.update(
        {
            "schema": "intent_alignment_verl_one_step_result_v3",
            "optimizer_steps_actual": 1,
            "source_artifact_sha256": sha256_file(raw_status_path),
            "source_training_log_sha256": sha256_file(log_source),
            "checkpoint_inventory_sha256": sha256_file(inventory_path),
            "hardware": {
                "device": contract["hardware"]["gpu"],
                "gpu_count": contract["hardware"]["gpu_count"],
                "reused_existing_allocation": True,
                "new_allocation_requested": False,
            },
        }
    )

    published_names_without_status = [
        "after-test100.json",
        "before-test100.json",
        "checkpoint-inventory.json",
        "launch.redacted.sh",
        "paired-comparison.json",
        "parameter-evidence.json",
        "resolved-config.redacted.yaml",
        "rollouts-step1.jsonl",
        "run-contract.redacted.json",
        "training-complete.json",
        "training-evidence.json",
        "training-log.redacted.txt",
    ]
    status["artifacts"] = {
        name: sha256_file(output_dir / name)
        for name in published_names_without_status
    }
    status_path = output_dir / "result-status.json"
    write_json(status_path, status)

    source_paths = {
        "after-test100.json": run_dir / "evaluation" / "after-test100.json",
        "before-test100.json": run_dir / "evaluation" / "before-test100.json",
        "checkpoint-inventory.json": inventory_path,
        "launch.redacted.sh": redacted_sources["launcher"][0],
        "paired-comparison.json": run_dir / "evaluation" / "paired-comparison.json",
        "parameter-evidence.json": run_dir / "parameter-evidence.json",
        "resolved-config.redacted.yaml": redacted_sources["resolved_config"][0],
        "rollouts-step1.jsonl": rollout_source,
        "run-contract.redacted.json": redacted_sources["run_contract"][0],
        "training-complete.json": run_dir / "training-complete.json",
        "training-evidence.json": run_dir / "training-evidence.json",
        "training-log.redacted.txt": log_source,
        "result-status.json": raw_status_path,
    }
    published_names = sorted(source_paths)
    manifest = {
        "schema": "intent_alignment_public_result_manifest_v2",
        "run_id": run_id,
        "source_commit": contract["source"]["commit"],
        "publication_note": (
            "Public copies preserve task outputs, rollout rows, metrics, and "
            "tensor probes while replacing machine-local paths, hostname, Pod "
            "IP, and platform task identity, removing ANSI control sequences, "
            "and trimming semantically empty trailing whitespace. Source and "
            "published hashes are labelled separately."
        ),
        "redacted_fields": [
            "evaluation/*.model.path",
            "evaluation/*.runtime.hostname",
            "parameter-evidence.base_path",
            "parameter-evidence.after_path",
            "training-log machine-local paths and Pod identity",
            "run-contract machine-local paths and Pod identity",
            "resolved-config machine-local paths",
            "launcher machine-local paths",
            "result-status.hardware_task",
        ],
        "source_artifact_sha256": {
            name: sha256_file(source_paths[name]) for name in published_names
        },
        "published_artifact_sha256": {
            name: sha256_file(output_dir / name) for name in published_names
        },
    }
    write_json(output_dir / "artifact-manifest.json", manifest)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "published_artifacts": len(published_names),
                "source_rollout_sha256": sha256_file(rollout_source),
                "published_rollout_sha256": sha256_file(rollout_public),
                "source_training_log_sha256": sha256_file(log_source),
                "published_training_log_sha256": sha256_file(log_public),
                "checkpoint_inventory_sha256": sha256_file(inventory_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
