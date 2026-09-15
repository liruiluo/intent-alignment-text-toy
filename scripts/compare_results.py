#!/usr/bin/env python3
"""Create a strict paired before/after report for frozen test100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from intent_alignment_toy.environment import Task, TextNavigationEnv
from intent_alignment_toy.evaluation import parse_action
from intent_alignment_toy.evidence import EXPECTED_CONFIG_NORMALIZATION

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
EXPECTED_BASE_MODEL_CONFIG_SHA256 = (
    "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
)
EXPECTED_MODEL_CONFIG_FINGERPRINT = {
    "class": "Qwen3_5Config",
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "sha256": "59a96f1b0661530c8c919d4529ece05245e106b30c2ac46e4a0f0c4be9f6cc78",
}
MODEL_IDENTITY_FIELDS = ("class", "model_type", "architectures")
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
    "model_class": "Qwen3_5ForConditionalGeneration",
}
RUNTIME_IDENTITY_FIELDS = tuple(EXPECTED_RUNTIME)
TASK_IDENTITY_FIELDS = ("instruction", "target", "max_steps")
EXPECTED_POLICY_CORE_TENSORS = 426
EXPECTED_BASE_MTP_TENSORS = 15
EXPECTED_VISION_TENSORS = 297
EXPECTED_PROBE_LAYERS = (0, 1, 8, 16, 24, 31)
EXPECTED_PROBE_NAMES = tuple(
    f"model.language_model.layers.{layer}.input_layernorm.weight"
    for layer in EXPECTED_PROBE_LAYERS
)
EXPECTED_PROBE_SHAPE = [2560]
EXPECTED_MIXED_REWARD_GROUPS = 66
EXPECTED_SOURCE_COMMIT = "100f0f4c117779f941cf38f61ef095f58c2ea1a6"
EXPECTED_VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"
EXPECTED_TRAIN_PARQUET_SHA256 = (
    "4ab5bfa4f55f073617795c403a10a1b45310d136691ea5dff0377a58cb991ae1"
)
EXPECTED_TEST_PARQUET_SHA256 = (
    "455a4c17c370bf40802a282215f69c71a5d21326ad185375211099bd6fe68730"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def weight_manifest(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    files = payload.get("model", {}).get("weights", {}).get("files", [])
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label}: weight manifest is empty")

    normalized: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise TypeError(f"{label}: weight manifest row is not an object")
        if set(item) != {"name", "bytes", "sha256", "shard_id"}:
            raise ValueError(f"{label}: weight manifest row schema drifted")
        name = item.get("name")
        size = item.get("bytes")
        digest = item.get("sha256")
        shard_id = item.get("shard_id")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}: invalid weight filename")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"{label}: invalid weight file size for {name!r}")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{label}: invalid weight SHA256 for {name!r}")
        if not isinstance(shard_id, str) or not shard_id:
            raise ValueError(f"{label}: invalid shard id for {name!r}")
        normalized.append(
            {"name": name, "bytes": size, "sha256": digest, "shard_id": shard_id}
        )

    names = [item["name"] for item in normalized]
    if len(set(names)) != len(names):
        raise ValueError(f"{label}: duplicate weight filenames")
    return sorted(normalized, key=lambda item: item["name"])


def replay_task_result(
    row: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    task_id = str(expected["id"])
    required_fields = {
        "id",
        "instruction",
        "target",
        "max_steps",
        "actions",
        "raw_outputs",
        "position",
        "steps_used",
        "remaining_steps",
        "terminated",
        "success",
        "reason",
        "trajectory",
    }
    if set(row) != required_fields:
        raise ValueError(f"{label}: {task_id} result schema drifted")
    for field in TASK_IDENTITY_FIELDS:
        if row.get(field) != expected.get(field):
            raise ValueError(f"{label}: {task_id} {field} drifted")
    if not isinstance(row.get("success"), bool):
        raise TypeError(f"{label}: {task_id} success is not boolean")
    actions = row.get("actions")
    raw_outputs = row.get("raw_outputs")
    if not isinstance(actions, list) or not all(isinstance(x, str) for x in actions):
        raise TypeError(f"{label}: {task_id} actions must be strings")
    if not isinstance(raw_outputs, list) or not all(
        isinstance(x, str) for x in raw_outputs
    ):
        raise TypeError(f"{label}: {task_id} raw outputs must be strings")

    task = Task(
        task_id=task_id,
        instruction=str(expected["instruction"]),
        target=tuple(int(value) for value in expected["target"]),
        max_steps=int(expected["max_steps"]),
    )
    env = TextNavigationEnv(task)
    normalized_actions: list[str] = []
    for raw_output in raw_outputs:
        if env.terminated:
            raise ValueError(f"{label}: {task_id} output continues after termination")
        action = parse_action(raw_output)
        if action is None:
            env.step("INVALID")
        else:
            normalized_actions.append(action)
            env.step(action)
    if not env.terminated:
        raise ValueError(f"{label}: {task_id} trajectory is not terminal")
    result = env.result()
    expected_result = {
        "position": list(result.position),
        "steps_used": result.steps_used,
        "remaining_steps": result.remaining_steps,
        "terminated": result.terminated,
        "success": result.success,
        "reason": result.reason,
        "trajectory": [list(position) for position in result.trajectory],
    }
    if normalized_actions != actions:
        raise ValueError(f"{label}: {task_id} normalized actions drifted")
    for field, value in expected_result.items():
        if row.get(field) != value:
            raise ValueError(f"{label}: {task_id} replay field {field!r} drifted")


def validate_payload(
    payload: dict[str, Any], label: str, *, require_base_config: bool
) -> dict[str, dict[str, Any]]:
    protocol = payload.get("protocol", {})
    expected_tasks = frozen_tasks()
    checks = {
        "protocol name": (protocol.get("name"), EXPECTED_PROTOCOL_NAME),
        "dataset name": (protocol.get("dataset"), FROZEN_TEST.name),
        "dataset hash": (protocol.get("dataset_sha256"), EXPECTED_DATASET_SHA256),
        "prompt contract": (protocol.get("prompt_contract"), EXPECTED_PROMPT_CONTRACT),
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
        replay_task_result(row, expected, label)

    summary = payload.get("summary", {})
    successes = sum(row["success"] for row in rows)
    expected_summary = {
        "successes": successes,
        "total": 100,
        "success_rate": successes / 100,
        "at_least_95_percent_requires": 95,
        "meets_at_least_95_percent": successes >= 95,
        "meets_strictly_above_95_percent": successes / 100 > 0.95,
    }
    if summary != expected_summary:
        raise ValueError(f"{label}: summary is inconsistent with replayed task rows")

    model = payload.get("model", {})
    if require_base_config:
        if model.get("config_sha256") != EXPECTED_BASE_MODEL_CONFIG_SHA256:
            raise ValueError(
                f"{label}: model config is not the frozen base Qwen3.5-4B config"
            )
        if model.get("config_fingerprint") != EXPECTED_MODEL_CONFIG_FINGERPRINT:
            raise ValueError(f"{label}: semantic model config fingerprint drifted")
    else:
        fingerprint = model.get("config_fingerprint", {})
        for field in MODEL_IDENTITY_FIELDS:
            if fingerprint.get(field) != EXPECTED_MODEL_CONFIG_FINGERPRINT[field]:
                raise ValueError(f"{label}: model identity field {field!r} drifted")
    if model.get("tokenizer_fingerprint") != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ValueError(f"{label}: tokenizer fingerprint drifted")

    runtime = payload.get("runtime", {})
    for field, expected in EXPECTED_RUNTIME.items():
        if runtime.get(field) != expected:
            raise ValueError(
                f"{label}: runtime field {field!r} drifted: "
                f"expected {expected!r}, got {runtime.get(field)!r}"
            )
    manifest = weight_manifest(payload, label)
    if require_base_config:
        expected_base_manifest = [
            {
                "bytes": 5_329_398_688,
                "name": "model.safetensors-00001-of-00002.safetensors",
                "sha256": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
                "shard_id": "00001-of-00002",
            },
            {
                "bytes": 3_990_429_408,
                "name": "model.safetensors-00002-of-00002.safetensors",
                "sha256": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
                "shard_id": "00002-of-00002",
            },
        ]
        if manifest != expected_base_manifest:
            raise ValueError(f"{label}: frozen base weight manifest drifted")
    return tasks


def validate_parameter_evidence(
    evidence: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if evidence.get("schema") != "intent_alignment_semantic_parameter_evidence_v2":
        raise ValueError("unsupported parameter evidence schema")
    if evidence.get("base_config_sha256") != before["model"]["config_sha256"]:
        raise ValueError("parameter evidence base config binding drifted")
    if evidence.get("after_config_sha256") != after["model"]["config_sha256"]:
        raise ValueError("parameter evidence trained config binding drifted")
    if evidence.get("allowed_config_normalization") != EXPECTED_CONFIG_NORMALIZATION:
        raise ValueError("parameter evidence config normalization drifted")

    expected_flags = {
        "policy_core_key_sets_equal": True,
        "weights_changed": True,
        "effective_bfloat16_weights_changed": True,
    }
    for field, expected in expected_flags.items():
        if evidence.get(field) is not expected:
            raise ValueError(f"parameter evidence field {field!r} is not {expected}")
    if evidence.get("policy_core_tensor_count") != EXPECTED_POLICY_CORE_TENSORS:
        raise ValueError("parameter evidence policy core tensor count drifted")
    if evidence.get("excluded_base_mtp_tensor_count") != EXPECTED_BASE_MTP_TENSORS:
        raise ValueError("parameter evidence MTP exclusion count drifted")
    if evidence.get("vision_tensor_count") != {
        "before": EXPECTED_VISION_TENSORS,
        "after": EXPECTED_VISION_TENSORS,
    }:
        raise ValueError("parameter evidence vision tensor count drifted")

    manifests = evidence.get("evaluated_weight_files")
    expected_manifests = {
        "before": weight_manifest(before, "before"),
        "after": weight_manifest(after, "after"),
    }
    if manifests != expected_manifests:
        raise ValueError("parameter evidence evaluated weight binding drifted")

    probes = evidence.get("probes", [])
    if evidence.get("probe_rule") != (
        "input_layernorm weights at layers "
        + ",".join(str(layer) for layer in EXPECTED_PROBE_LAYERS)
    ):
        raise ValueError("parameter evidence probe rule drifted")
    if tuple(row.get("name") for row in probes) != EXPECTED_PROBE_NAMES:
        raise ValueError("parameter evidence probe identities drifted")
    if evidence.get("changed_probe_count") != len(EXPECTED_PROBE_NAMES):
        raise ValueError("parameter evidence changed probe count is inconsistent")

    required_probe_fields = {
        "name",
        "shape",
        "base_dtype",
        "after_dtype",
        "base_float32_sha256",
        "after_float32_sha256",
        "changed_float32_values",
        "effective_bfloat16_changed_values",
        "max_abs_delta",
        "mean_abs_delta",
    }
    for row in probes:
        if set(row) != required_probe_fields:
            raise ValueError(f"parameter probe {row.get('name')!r} schema drifted")
        if row.get("shape") != EXPECTED_PROBE_SHAPE:
            raise ValueError(f"parameter probe {row['name']!r} shape drifted")
        if row.get("base_dtype") != "torch.bfloat16":
            raise ValueError(f"parameter probe {row['name']!r} base dtype drifted")
        if row.get("after_dtype") != "torch.float32":
            raise ValueError(f"parameter probe {row['name']!r} trained dtype drifted")
        for field in ("base_float32_sha256", "after_float32_sha256"):
            digest = row.get(field)
            if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"parameter probe {row['name']!r} hash drifted")
        if row["base_float32_sha256"] == row["after_float32_sha256"]:
            raise ValueError(f"parameter probe {row['name']!r} did not change")
        changed = row.get("changed_float32_values")
        bf16_changed = row.get("effective_bfloat16_changed_values")
        if not isinstance(changed, int) or isinstance(changed, bool) or not 1 <= changed <= 2560:
            raise ValueError(f"parameter probe {row['name']!r} FP32 count drifted")
        if (
            not isinstance(bf16_changed, int)
            or isinstance(bf16_changed, bool)
            or not 0 <= bf16_changed <= changed
        ):
            raise ValueError(f"parameter probe {row['name']!r} BF16 count drifted")
        maximum = row.get("max_abs_delta")
        mean = row.get("mean_abs_delta")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0
            for value in (maximum, mean)
        ) or float(maximum) < float(mean):
            raise ValueError(f"parameter probe {row['name']!r} delta drifted")

    effective_changes = sum(
        int(row["effective_bfloat16_changed_values"]) for row in probes
    )
    if effective_changes <= 0:
        raise ValueError("parameter evidence has no effective BF16 change")
    if evidence.get("effective_bfloat16_changed_value_count") != effective_changes:
        raise ValueError("parameter evidence BF16 change count is inconsistent")

    serialization = evidence.get("serialization", {})
    if sorted(serialization.get("before_files", [])) != sorted(
        item["name"] for item in expected_manifests["before"]
    ):
        raise ValueError("parameter evidence base serialization does not match evaluation")
    if sorted(serialization.get("after_files", [])) != sorted(
        item["name"] for item in expected_manifests["after"]
    ):
        raise ValueError(
            "parameter evidence trained serialization does not match evaluation"
        )


def _rollout_group_id(uid: str) -> tuple[str, int, int]:
    parts = uid.rsplit("_", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"invalid rollout uid {uid!r}")
    try:
        rollout_index = int(parts[1])
        turn_index = int(parts[2])
    except ValueError as error:
        raise ValueError(f"invalid rollout uid {uid!r}") from error
    return parts[0], rollout_index, turn_index


def validate_rollout_artifact(path: Path, evidence: dict[str, Any]) -> None:
    if sha256_file(path) != evidence.get("published_rollout_sha256"):
        raise ValueError("training rollout SHA256 drifted")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != evidence.get("trajectory_count"):
        raise ValueError("training rollout trajectory count drifted")
    groups: dict[str, list[tuple[int, float, str, str]]] = {}
    for row in rows:
        if set(row) != {"gts", "input", "output", "score", "step", "uid"}:
            raise ValueError("training rollout row schema drifted")
        if row["step"] != evidence.get("global_step"):
            raise ValueError("training rollout global step drifted")
        if not isinstance(row["uid"], str):
            raise TypeError("training rollout uid must be a string")
        group_id, rollout_index, turn_index = _rollout_group_id(row["uid"])
        if rollout_index not in (0, 1) or turn_index != 0:
            raise ValueError("training rollout uid suffix drifted")
        score = row["score"]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or float(score) not in (0.0, 1.0)
        ):
            raise ValueError("training rollout score drifted")
        for field in ("gts", "input", "output"):
            if not isinstance(row[field], str):
                raise TypeError(f"training rollout field {field!r} must be a string")
        groups.setdefault(group_id, []).append(
            (rollout_index, float(score), row["input"], row["gts"])
        )

    if len(groups) != evidence.get("grpo_group_count"):
        raise ValueError("training rollout GRPO group count drifted")
    expected_rollouts = evidence.get("rollouts_per_group")
    mixed = 0
    reward_values: set[float] = set()
    for group_rows in groups.values():
        if len(group_rows) != expected_rollouts:
            raise ValueError("training rollout group cardinality drifted")
        if sorted(row[0] for row in group_rows) != list(range(expected_rollouts)):
            raise ValueError("training rollout group indices drifted")
        if len({row[2] for row in group_rows}) != 1 or len(
            {row[3] for row in group_rows}
        ) != 1:
            raise ValueError("training rollout group task identity drifted")
        scores = {row[1] for row in group_rows}
        reward_values.update(scores)
        mixed += len(scores) > 1
    if sorted(reward_values) != evidence.get("reward_values"):
        raise ValueError("training rollout reward values drifted")
    if mixed != evidence.get("mixed_reward_group_count"):
        raise ValueError("training rollout mixed-group count drifted")


def validate_training_log(path: Path, evidence: dict[str, Any]) -> None:
    if sha256_file(path) != evidence.get("published_training_log_sha256"):
        raise ValueError("published training log SHA256 drifted")
    text = path.read_text(encoding="utf-8")
    step_values = [
        int(value) for value in re.findall(r"training/global_step:([0-9]+)", text)
    ]
    if sorted(set(step_values)) != [evidence.get("global_step")]:
        raise ValueError("training log global-step evidence drifted")
    grad_norms = [
        float(value)
        for value in re.findall(
            r"actor/grad_norm:([-+0-9.eE]+)",
            text,
        )
    ]
    if grad_norms != evidence.get("positive_finite_actor_grad_norm"):
        raise ValueError("training log actor gradient evidence drifted")
    if not all(math.isfinite(value) and value > 0 for value in grad_norms):
        raise ValueError("training log actor gradient is not positive finite")
    if not re.search(r"Training Progress: 100%\|[^\n]*\b1/1\b", text):
        raise ValueError("training log does not prove one completed update")


def validate_checkpoint_inventory(
    inventory: dict[str, Any], evidence: dict[str, Any], after: dict[str, Any]
) -> None:
    if inventory.get("schema") != "intent_alignment_checkpoint_inventory_v1":
        raise ValueError("unsupported checkpoint inventory schema")
    if inventory.get("global_step") != 1:
        raise ValueError("checkpoint inventory global step drifted")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint inventory is empty")
    by_name: dict[str, dict[str, Any]] = {}
    for row in files:
        if set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("checkpoint inventory row schema drifted")
        path = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        if not isinstance(path, str) or not path.startswith("global_step_1/"):
            raise ValueError("checkpoint inventory path drifted")
        if path in by_name:
            raise ValueError("checkpoint inventory contains duplicate paths")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("checkpoint inventory size drifted")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError("checkpoint inventory SHA256 drifted")
        by_name[path] = row
    required_suffixes = {
        "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt",
        "actor/extra_state_world_size_1_rank_0.pt",
        "data.pt",
        "actor/huggingface/model.safetensors",
    }
    found_suffixes = {path.removeprefix("global_step_1/") for path in by_name}
    if not required_suffixes <= found_suffixes:
        raise ValueError("checkpoint inventory misses required training state")
    after_manifest = weight_manifest(after, "after")
    for row in after_manifest:
        checkpoint_path = f"global_step_1/actor/huggingface/{row['name']}"
        checkpoint_row = by_name.get(checkpoint_path)
        if (
            checkpoint_row is None
            or checkpoint_row["bytes"] != row["bytes"]
            or checkpoint_row["sha256"] != row["sha256"]
        ):
            raise ValueError("checkpoint inventory does not bind evaluated weights")


def validate_training_evidence(
    evidence: dict[str, Any],
    *,
    rollout_path: Path | None = None,
    training_log_path: Path | None = None,
    checkpoint_inventory_path: Path | None = None,
    run_contract_path: Path | None = None,
    resolved_config_path: Path | None = None,
    launcher_path: Path | None = None,
    after: dict[str, Any] | None = None,
) -> None:
    if evidence.get("schema") != "intent_alignment_one_step_training_evidence_v2":
        raise ValueError("unsupported training evidence schema")
    expected = {
        "global_step": 1,
        "optimizer_updates_expected": 1,
        "optimizer_updates_actual": 1,
        "trajectory_count": 800,
        "grpo_group_count": 400,
        "rollouts_per_group": 2,
        "mixed_reward_group_count": EXPECTED_MIXED_REWARD_GROUPS,
        "reward_values": [0.0, 1.0],
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "verl_commit": EXPECTED_VERL_COMMIT,
        "train_parquet_sha256": EXPECTED_TRAIN_PARQUET_SHA256,
        "test_parquet_sha256": EXPECTED_TEST_PARQUET_SHA256,
        "test_jsonl_sha256": EXPECTED_DATASET_SHA256,
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ValueError(f"training evidence field {field!r} drifted")
    for field in (
        "source_training_log_sha256",
        "published_training_log_sha256",
        "source_rollout_sha256",
        "published_rollout_sha256",
        "checkpoint_inventory_sha256",
        "source_run_contract_sha256",
        "published_run_contract_sha256",
        "source_resolved_config_sha256",
        "published_resolved_config_sha256",
        "source_launcher_sha256",
        "published_launcher_sha256",
    ):
        digest = evidence.get(field)
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"training evidence field {field!r} is not a SHA256")
    grad_norms = evidence.get("positive_finite_actor_grad_norm", [])
    if (
        not isinstance(grad_norms, list)
        or len(grad_norms) != 1
        or not math.isfinite(float(grad_norms[0]))
        or float(grad_norms[0]) <= 0
        or evidence.get("actor_grad_norm_values") != grad_norms
    ):
        raise ValueError("training evidence has invalid actor gradient norms")
    if rollout_path is not None:
        if evidence.get("rollout_file") != rollout_path.name:
            raise ValueError("training rollout filename binding drifted")
        validate_rollout_artifact(rollout_path, evidence)
    if training_log_path is not None:
        if evidence.get("training_log_file") != training_log_path.name:
            raise ValueError("training log filename binding drifted")
        validate_training_log(training_log_path, evidence)
    if checkpoint_inventory_path is not None:
        if after is None:
            raise ValueError("trained evaluation payload is required for checkpoint audit")
        if evidence.get("checkpoint_inventory_file") != checkpoint_inventory_path.name:
            raise ValueError("checkpoint inventory filename binding drifted")
        validate_checkpoint_inventory(load(checkpoint_inventory_path), evidence, after)
    bound_files = (
        (
            run_contract_path,
            "run_contract_file",
            "published_run_contract_sha256",
            "run contract",
        ),
        (
            resolved_config_path,
            "resolved_config_file",
            "published_resolved_config_sha256",
            "resolved config",
        ),
        (launcher_path, "launcher_file", "published_launcher_sha256", "launcher"),
    )
    for path, name_field, hash_field, label in bound_files:
        if path is None:
            continue
        if evidence.get(name_field) != path.name:
            raise ValueError(f"{label} filename binding drifted")
        if sha256_file(path) != evidence.get(hash_field):
            raise ValueError(f"{label} SHA256 drifted")


def build_report(
    before: dict[str, Any],
    after: dict[str, Any],
    parameter_evidence: dict[str, Any],
    training_evidence: dict[str, Any],
    before_path: Path,
    after_path: Path,
    parameter_evidence_path: Path,
    training_evidence_path: Path,
    rollout_path: Path | None = None,
    training_log_path: Path | None = None,
    checkpoint_inventory_path: Path | None = None,
    run_contract_path: Path | None = None,
    resolved_config_path: Path | None = None,
    launcher_path: Path | None = None,
) -> dict[str, Any]:
    before_tasks = validate_payload(before, "before", require_base_config=True)
    after_tasks = validate_payload(after, "after", require_base_config=False)

    if before["protocol"] != after["protocol"]:
        raise ValueError("before/after protocol objects differ")
    if (
        before["model"]["tokenizer_fingerprint"]
        != after["model"]["tokenizer_fingerprint"]
    ):
        raise ValueError("before/after tokenizer fingerprints differ")
    for field in MODEL_IDENTITY_FIELDS:
        if (
            before["model"]["config_fingerprint"][field]
            != after["model"]["config_fingerprint"][field]
        ):
            raise ValueError(f"before/after model identity field {field!r} differs")
    for field in RUNTIME_IDENTITY_FIELDS:
        if before["runtime"][field] != after["runtime"][field]:
            raise ValueError(f"before/after runtime field {field!r} differs")
    validate_parameter_evidence(parameter_evidence, before, after)
    if parameter_evidence.get("checkpoint_inventory_sha256") != training_evidence.get(
        "checkpoint_inventory_sha256"
    ):
        raise ValueError("parameter/training checkpoint inventory binding drifted")
    if (
        rollout_path is None
        or training_log_path is None
        or checkpoint_inventory_path is None
        or run_contract_path is None
        or resolved_config_path is None
        or launcher_path is None
    ):
        raise ValueError(
            "rollout, training log, checkpoint inventory, contract, config, and "
            "launcher are required"
        )
    if sha256_file(checkpoint_inventory_path) != training_evidence.get(
        "checkpoint_inventory_sha256"
    ):
        raise ValueError("checkpoint inventory artifact SHA256 drifted")
    validate_training_evidence(
        training_evidence,
        rollout_path=rollout_path,
        training_log_path=training_log_path,
        checkpoint_inventory_path=checkpoint_inventory_path,
        run_contract_path=run_contract_path,
        resolved_config_path=resolved_config_path,
        launcher_path=launcher_path,
        after=after,
    )

    paired = {
        "improved": [],
        "regressed": [],
        "unchanged_success": [],
        "unchanged_failure": [],
    }
    pair_labels = {
        (False, True): "improved",
        (True, False): "regressed",
        (True, True): "unchanged_success",
        (False, False): "unchanged_failure",
    }
    for task_id in before_tasks:
        for field in TASK_IDENTITY_FIELDS:
            if before_tasks[task_id][field] != after_tasks[task_id][field]:
                raise ValueError(f"before/after {task_id} {field} differs")
        pair = (before_tasks[task_id]["success"], after_tasks[task_id]["success"])
        paired[pair_labels[pair]].append(task_id)

    return {
        "schema": "intent_alignment_paired_comparison_v2",
        "before_result": before_path.name,
        "after_result": after_path.name,
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
        "paired": paired,
        "evaluation_identity": {
            "tokenizer_fingerprint": before["model"]["tokenizer_fingerprint"],
            "base_model_config_fingerprint": before["model"]["config_fingerprint"],
            "trained_model_config_fingerprint": after["model"]["config_fingerprint"],
            "config_normalization": parameter_evidence[
                "allowed_config_normalization"
            ],
            "runtime": {
                field: before["runtime"][field] for field in RUNTIME_IDENTITY_FIELDS
            },
        },
        "training_evidence": {
            "path": training_evidence_path.name,
            "sha256": sha256_file(training_evidence_path),
            "global_step": training_evidence["global_step"],
            "optimizer_updates_actual": training_evidence[
                "optimizer_updates_actual"
            ],
            "trajectory_count": training_evidence["trajectory_count"],
            "grpo_group_count": training_evidence["grpo_group_count"],
            "mixed_reward_group_count": training_evidence[
                "mixed_reward_group_count"
            ],
            "actor_grad_norm": training_evidence[
                "positive_finite_actor_grad_norm"
            ][0],
        },
        "parameter_evidence": {
            "path": parameter_evidence_path.name,
            "sha256": sha256_file(parameter_evidence_path),
            "policy_core_tensor_count": parameter_evidence[
                "policy_core_tensor_count"
            ],
            "changed_probe_count": parameter_evidence["changed_probe_count"],
            "effective_bfloat16_changed_value_count": parameter_evidence[
                "effective_bfloat16_changed_value_count"
            ],
            "weights_changed": True,
            "evaluated_weight_files": parameter_evidence[
                "evaluated_weight_files"
            ],
        },
        "audit_artifacts": {
            "rollout": {
                "path": rollout_path.name,
                "sha256": sha256_file(rollout_path),
            },
            "training_log": {
                "path": training_log_path.name,
                "sha256": sha256_file(training_log_path),
            },
            "checkpoint_inventory": {
                "path": checkpoint_inventory_path.name,
                "sha256": sha256_file(checkpoint_inventory_path),
            },
            "run_contract": {
                "path": run_contract_path.name,
                "sha256": sha256_file(run_contract_path),
            },
            "resolved_config": {
                "path": resolved_config_path.name,
                "sha256": sha256_file(resolved_config_path),
            },
            "launcher": {
                "path": launcher_path.name,
                "sha256": sha256_file(launcher_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--parameter-evidence", type=Path, required=True)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        load(args.before),
        load(args.after),
        load(args.parameter_evidence),
        load(args.training_evidence),
        args.before,
        args.after,
        args.parameter_evidence,
        args.training_evidence,
        args.rollout,
        args.training_log,
        args.checkpoint_inventory,
        args.run_contract,
        args.resolved_config,
        args.launcher,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
