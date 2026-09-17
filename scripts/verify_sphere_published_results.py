#!/usr/bin/env python3
"""Verify the lightweight public evidence for the three one-step SPHERE arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "results" / "qwen3.5-4b-sphere-one-step-v1"
EXPECTED_VARIANTS = {
    "sphere_dense_last_adaptive_v1": {
        "layer_mode": "last",
        "target_scale_mode": "adaptive_trace",
        "fixed_alpha": None,
        "target_count": 1,
        "target_layer_indices": [31],
    },
    "sphere_dense_all_hidden_adaptive_v1": {
        "layer_mode": "all_hidden",
        "target_scale_mode": "adaptive_trace",
        "fixed_alpha": None,
        "target_count": 32,
        "target_layer_indices": list(range(32)),
    },
    "sphere_dense_last_fixed_alpha_v1": {
        "layer_mode": "last",
        "target_scale_mode": "fixed",
        "fixed_alpha": 1.0,
        "target_count": 1,
        "target_layer_indices": [31],
    },
}
EXPECTED_TEST_SHA256 = (
    "b2f7560c4466949136bec09bd2e8532a27def45696c5eeed3b4fcc99de2c1675"
)
EXPECTED_VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRIVATE_MARKERS = (
    "/home/",
    "/Users/",
    "/dev/shm",
    "6.235.",
    "luolirui-1-",
    "master-0",
    "D:\\",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid SHA256 for {label}: {value!r}")
    return value


def verify(result_dir: Path) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    publication = load_json(result_dir / "publication-manifest.json")
    if publication.get("schema") != "intent_alignment_sphere_publication_manifest_v1":
        raise ValueError("unsupported SPHERE publication manifest schema")
    published = publication.get("artifacts")
    if not isinstance(published, dict) or not published:
        raise ValueError("SPHERE publication manifest is empty")
    for relative, expected in published.items():
        require_sha256(expected, f"published artifact {relative}")
        path = result_dir / relative
        if not path.is_file():
            raise ValueError(f"published artifact is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"published artifact hash drifted: {relative}")

    actual_jsons = {
        str(path.relative_to(result_dir))
        for path in result_dir.rglob("*.json")
        if path.name != "publication-manifest.json"
    }
    if actual_jsons != set(published):
        raise ValueError("publication manifest file set drifted")

    for path in result_dir.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker in text:
                raise ValueError(f"{path.relative_to(result_dir)} contains {marker!r}")

    summary = load_json(result_dir / "summary.json")
    if summary.get("schema") != "intent_alignment_sphere_three_arm_summary_v1":
        raise ValueError("unsupported SPHERE summary schema")
    if summary.get("framework") != {
        "name": "veRL",
        "version": "v0.9.0",
        "commit": EXPECTED_VERL_COMMIT,
    }:
        raise ValueError("veRL identity drifted")
    if summary.get("model") != "Qwen3.5-4B":
        raise ValueError("model identity drifted")
    if summary.get("dataset", {}).get("test_jsonl_sha256") != EXPECTED_TEST_SHA256:
        raise ValueError("test dataset identity drifted")
    if summary.get("dataset", {}).get("train") != 400:
        raise ValueError("training set size drifted")
    if summary.get("dataset", {}).get("test") != 100:
        raise ValueError("test set size drifted")
    common = summary.get("common_training", {})
    if common.get("optimizer_updates") != 1 or common.get("global_step") != 1:
        raise ValueError("summary does not describe exactly one update")
    if common.get("trajectories") != 800 or common.get("grpo_groups") != 400:
        raise ValueError("summary rollout cardinality drifted")
    if common.get("rollout_n") != 2 or common.get("thinking") is not False:
        raise ValueError("summary rollout/thinking contract drifted")

    summary_arms = {arm["variant"]: arm for arm in summary.get("arms", [])}
    if set(summary_arms) != set(EXPECTED_VARIANTS):
        raise ValueError("SPHERE arm set drifted")

    mixed_groups: dict[str, int] = {}
    for variant, expected_variant in EXPECTED_VARIANTS.items():
        arm_dir = result_dir / variant
        manifest = load_json(arm_dir / "artifact-manifest.json")
        if manifest.get("schema") != "intent_alignment_sphere_public_arm_manifest_v1":
            raise ValueError(f"{variant}: invalid arm manifest schema")
        if manifest.get("variant") != variant:
            raise ValueError(f"{variant}: manifest identity drifted")
        source_hashes = manifest.get("source_artifact_sha256", {})
        public_hashes = manifest.get("published_artifact_sha256", {})
        if set(source_hashes) != set(public_hashes):
            raise ValueError(f"{variant}: source/public manifest file sets differ")
        for name, expected_hash in public_hashes.items():
            require_sha256(expected_hash, f"{variant}/{name}")
            if sha256_file(arm_dir / name) != expected_hash:
                raise ValueError(f"{variant}: public hash drifted for {name}")
        for name, digest in source_hashes.items():
            require_sha256(digest, f"{variant} source {name}")

        status = load_json(arm_dir / "result-status.json")
        training = load_json(arm_dir / "training-evidence.json")
        complete = load_json(arm_dir / "training-complete.json")
        worker = load_json(arm_dir / "worker-evidence.json")
        checkpoint = load_json(arm_dir / "checkpoint-inventory.json")
        pipeline = load_json(arm_dir / "pipeline-exit.json")
        parameter = load_json(arm_dir / "parameter-evidence.json")
        after = load_json(arm_dir / "after-test100.json")
        contract_public = load_json(arm_dir / "run-contract.json")
        arm = summary_arms[variant]

        if not all(
            payload.get("variant") == variant
            for payload in (status, training, complete, worker, pipeline)
        ):
            raise ValueError(f"{variant}: cross-artifact variant drifted")
        run_id = status.get("run_id")
        if not all(
            payload.get("run_id") == run_id
            for payload in (training, complete, checkpoint, pipeline)
        ) or worker.get("run", {}).get("id") != run_id:
            raise ValueError(f"{variant}: cross-artifact run identity drifted")
        if manifest.get("run_id") != run_id or arm.get("run_id") != run_id:
            raise ValueError(f"{variant}: publication run identity drifted")
        if contract_public.get("run_id") != run_id or contract_public.get("variant") != variant:
            raise ValueError(f"{variant}: public run contract identity drifted")
        if contract_public.get("runtime", {}).get("verl_commit") != EXPECTED_VERL_COMMIT:
            raise ValueError(f"{variant}: run contract veRL identity drifted")
        if contract_public.get("source", {}).get("base_commit") != manifest.get("source_commit"):
            raise ValueError(f"{variant}: source commit binding drifted")
        if contract_public.get("source", {}).get("manifest_sha256") != manifest.get("source_manifest_sha256"):
            raise ValueError(f"{variant}: source manifest binding drifted")
        if arm.get("source_commit") != manifest.get("source_commit"):
            raise ValueError(f"{variant}: summary source commit drifted")
        if arm.get("source_manifest_sha256") != manifest.get("source_manifest_sha256"):
            raise ValueError(f"{variant}: summary source manifest drifted")
        if contract_public.get("hardware", {}).get("gpu_count") != 1:
            raise ValueError(f"{variant}: run contract is not single-GPU")
        if contract_public.get("hardware", {}).get("new_allocation_requested") is not False:
            raise ValueError(f"{variant}: run contract requested a new allocation")
        if contract_public.get("constraints", {}).get("no_new_allocation") is not True:
            raise ValueError(f"{variant}: no-new-allocation constraint is absent")
        algorithm = contract_public.get("algorithm", {})
        if algorithm.get("sphere_variant") != variant:
            raise ValueError(f"{variant}: run contract algorithm identity drifted")
        if algorithm.get("trajectory_count") != 800 or algorithm.get("rollout_n") != 2:
            raise ValueError(f"{variant}: run contract rollout cardinality drifted")
        if contract_public.get("evaluation", {}).get("test_generation_only_after_training_endpoint") is not True:
            raise ValueError(f"{variant}: post-training evaluation ordering drifted")

        status_source_keys = {
            "after-test100.json": "after_test100_sha256",
            "checkpoint-inventory.json": "checkpoint_inventory_sha256",
            "parameter-evidence.json": "parameter_evidence_sha256",
            "run-contract.json": "run_contract_sha256",
            "training-complete.json": "training_complete_sha256",
            "training-evidence.json": "training_evidence_sha256",
            "worker-evidence.json": "worker_evidence_sha256",
        }
        for public_name, status_key in status_source_keys.items():
            if status.get("artifacts", {}).get(status_key) != source_hashes[public_name]:
                raise ValueError(f"{variant}: status/source hash drifted for {public_name}")

        if status.get("status") != "complete" or worker.get("status") != "passed":
            raise ValueError(f"{variant}: run did not complete successfully")
        if pipeline.get("exit_code") != 0:
            raise ValueError(f"{variant}: pipeline exit is nonzero")
        if not worker.get("checks") or not all(worker["checks"].values()):
            raise ValueError(f"{variant}: worker checks are incomplete")
        passes = worker.get("passes", {})
        if passes.get("optimizer_steps") != 1:
            raise ValueError(f"{variant}: worker did not execute exactly one optimizer step")
        if passes.get("ppo_only_forward_backward") != 1:
            raise ValueError(f"{variant}: PPO gradient pass count drifted")
        if passes.get("sphere_only_forward_backward") != 1:
            raise ValueError(f"{variant}: SPHERE gradient pass count drifted")
        if passes.get("combined_forward_backward") != 0:
            raise ValueError(f"{variant}: unexpected combined backward pass")
        if passes.get("explicit_gradient_combination") != 1:
            raise ValueError(f"{variant}: explicit gradient assembly is absent")

        if status.get("global_step") != 1 or status.get("optimizer_updates_actual") != 1:
            raise ValueError(f"{variant}: status is not exactly one update")
        if training.get("global_step") != 1 or training.get("optimizer_updates_actual") != 1:
            raise ValueError(f"{variant}: training evidence is not exactly one update")
        if complete.get("global_step") != 1 or complete.get("optimizer_updates_actual") != 1:
            raise ValueError(f"{variant}: completion receipt is not exactly one update")
        if checkpoint.get("global_step") != 1 or len(checkpoint.get("files", [])) < 1:
            raise ValueError(f"{variant}: checkpoint inventory is invalid")
        if training.get("trajectory_count") != 800:
            raise ValueError(f"{variant}: trajectory count drifted")
        if training.get("grpo_group_count") != 400:
            raise ValueError(f"{variant}: GRPO group count drifted")
        if training.get("rollouts_per_group") != 2:
            raise ValueError(f"{variant}: rollout.n drifted")
        mixed_groups[variant] = training.get("mixed_reward_group_count")

        if status.get("before_successes") != status.get("before_total") or status.get("before_total") != 100:
            raise ValueError(f"{variant}: pre-training test100 is not 100/100")
        if status.get("after_successes") != status.get("after_total") or status.get("after_total") != 100:
            raise ValueError(f"{variant}: post-training test100 is not 100/100")
        tasks = after.get("tasks", [])
        if len(tasks) != 100 or len({row.get("id") for row in tasks}) != 100:
            raise ValueError(f"{variant}: test100 task set is malformed")
        if sum(bool(row.get("success")) for row in tasks) != 100:
            raise ValueError(f"{variant}: test100 trajectories do not recompute to 100/100")
        if after.get("summary", {}).get("successes") != 100:
            raise ValueError(f"{variant}: evaluation summary drifted")
        if after.get("protocol", {}).get("dataset_sha256") != EXPECTED_TEST_SHA256:
            raise ValueError(f"{variant}: evaluation dataset hash drifted")
        if after.get("source_artifact_sha256") != source_hashes["after-test100.json"]:
            raise ValueError(f"{variant}: evaluation source binding drifted")

        target = worker.get("targets", {})
        contract = worker.get("contract", {})
        if target.get("count") != expected_variant["target_count"]:
            raise ValueError(f"{variant}: target count drifted")
        layer_indices = status.get("sphere", {}).get("target_layer_indices")
        if layer_indices != expected_variant["target_layer_indices"]:
            raise ValueError(f"{variant}: target layer indices drifted")
        for name in ("layer_mode", "target_scale_mode", "fixed_alpha"):
            if contract.get(name) != expected_variant[name]:
                raise ValueError(f"{variant}: {name} drifted")
        sphere = status.get("sphere", {})
        if not math.isclose(sphere.get("regularizer_to_ppo_norm_ratio"), 0.001):
            raise ValueError(f"{variant}: gradient norm ratio drifted")
        if sphere.get("postcast_regularizer_delta_l2_norm", 0.0) <= 0:
            raise ValueError(f"{variant}: SPHERE did not reach optimizer gradients")
        if sphere.get("full_tensor_assembly_residual_l2_norm") != 0.0:
            raise ValueError(f"{variant}: optimizer gradient assembly residual is nonzero")
        replay = sphere.get("replays", {}).get("sphere", {})
        if replay.get("identity_matches") != replay.get("microbatches"):
            raise ValueError(f"{variant}: replay identity count drifted")
        if replay.get("feature_relative_l2_difference", 1.0) > 0.05:
            raise ValueError(f"{variant}: replay relative L2 exceeds frozen tolerance")
        if replay.get("feature_relative_max_difference", 1.0) > 0.1:
            raise ValueError(f"{variant}: replay relative max exceeds frozen tolerance")
        if replay.get("raw_surrogate_relative_difference", 1.0) > 0.001:
            raise ValueError(f"{variant}: replay surrogate drift exceeds frozen tolerance")
        if parameter.get("weights_changed") is not True:
            raise ValueError(f"{variant}: FP32 parameter change is absent")
        if parameter.get("effective_bfloat16_weights_changed") is not True:
            raise ValueError(f"{variant}: BF16-effective parameter change is absent")

        if arm.get("variant") != variant or arm.get("pipeline_exit_code") != 0:
            raise ValueError(f"{variant}: top-level summary drifted")
        if arm.get("after_successes") != 100 or arm.get("optimizer_updates_actual") != 1:
            raise ValueError(f"{variant}: top-level result summary drifted")
        if arm.get("mixed_reward_group_count") != mixed_groups[variant]:
            raise ValueError(f"{variant}: mixed-reward summary drifted")
        if arm.get("source_result_status_sha256") != source_hashes["result-status.json"]:
            raise ValueError(f"{variant}: source result binding drifted")
        if arm.get("public_result_status_sha256") != public_hashes["result-status.json"]:
            raise ValueError(f"{variant}: public result binding drifted")

    if summary.get("all_arms_passed") is not True:
        raise ValueError("top-level all-arms verdict is not PASS")
    return {
        "status": "PASS",
        "arms": len(EXPECTED_VARIANTS),
        "test_successes_each": 100,
        "optimizer_updates_each": 1,
        "trajectories_each": 800,
        "mixed_reward_groups": mixed_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    print(json.dumps(verify(args.result_dir), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
