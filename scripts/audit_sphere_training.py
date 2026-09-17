#!/usr/bin/env python3
"""Fail-closed audits for dense-SPHERE one-update veRL runs.

The training audit derives the optimizer/global step from the real checkpoint
layout, hashes every checkpoint file, validates the semantic parameter-delta
evidence, checks the resolved veRL configuration, and binds all artifacts into a
training-complete receipt.  The final audit is intentionally separate and can
only run after that receipt exists; it then validates the frozen test100 result.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from intent_alignment_toy.sphere_contract import (
    SPHERE_REPLAY_RELATIVE_L2_TOLERANCE,
    SPHERE_REPLAY_RELATIVE_MAX_TOLERANCE,
    SPHERE_REPLAY_SURROGATE_RELATIVE_TOLERANCE,
    load_sphere_variant,
)

WORKER_SCHEMA = "intent_alignment_dense_sphere_worker_evidence_v4"
TRAINING_SCHEMA = "intent_alignment_dense_sphere_training_evidence_v3"
CHECKPOINT_SCHEMA = "intent_alignment_checkpoint_inventory_v1"
COMPLETION_SCHEMA = "intent_alignment_dense_sphere_training_complete_v3"
RESULT_SCHEMA = "intent_alignment_dense_sphere_result_v3"
PARAMETER_SCHEMA = "intent_alignment_semantic_parameter_evidence_v2"
RUN_CONTRACT_SCHEMA = "intent_alignment_dense_sphere_run_contract_v3"
EXPECTED_VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"
EXPECTED_TEST_JSONL_SHA256 = (
    "b2f7560c4466949136bec09bd2e8532a27def45696c5eeed3b4fcc99de2c1675"
)
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj$")
_STEP_RE = re.compile(r"^global_step_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object at {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _finite_positive(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return number


def _finite_nonnegative(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number < 0:
        raise ValueError(f"{label} must be non-negative, got {value!r}")
    return number


def _require_close(
    actual: Any,
    expected: Any,
    label: str,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> float:
    actual_number = _finite(actual, label)
    expected_number = _finite(expected, f"expected {label}")
    if not math.isclose(
        actual_number, expected_number, rel_tol=rel_tol, abs_tol=abs_tol
    ):
        raise ValueError(
            f"{label} drifted: actual={actual_number}, expected={expected_number}"
        )
    return actual_number


def _canonical_layer_indices(names: list[str]) -> list[int]:
    indices = []
    for name in names:
        canonical = ".".join(
            part for part in name.split(".") if part != "_fsdp_wrapped_module"
        )
        match = _LAYER_RE.fullmatch(canonical)
        if match is None:
            raise ValueError(f"non-canonical SPHERE target layer: {name}")
        indices.append(int(match.group(1)))
    return indices


def _validate_gradient_row(row: dict[str, Any], label: str) -> tuple[float, float]:
    norm = _finite_positive(row.get("l2_norm"), f"{label} gradient norm")
    projection = _finite(row.get("projection"), f"{label} gradient projection")
    if int(row.get("tensors_with_grad", 0)) <= 0:
        raise ValueError(f"{label} gradient tensor count is zero")
    if int(row.get("projection_sample_count", 0)) <= 0:
        raise ValueError(f"{label} gradient projection sample count is zero")
    return norm, projection


def validate_worker_evidence(
    payload: dict[str, Any],
    *,
    expected_variant: str,
    registry_path: Path,
    expected_run_id: str | None = None,
    expected_run_nonce: str | None = None,
    expected_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Validate worker evidence and recompute all SPHERE coefficient checks."""
    variant = load_sphere_variant(registry_path, expected_variant)
    if payload.get("schema") != WORKER_SCHEMA:
        raise ValueError("unexpected SPHERE worker evidence schema")
    if payload.get("status") != "passed":
        raise ValueError("SPHERE worker evidence is not a passed artifact")
    if payload.get("variant") != expected_variant:
        raise ValueError("SPHERE worker evidence variant drifted")
    run_identity = payload.get("run", {})
    run_id = run_identity.get("id")
    run_nonce = run_identity.get("nonce")
    evidence_path = run_identity.get("evidence_path")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(run_nonce, str)
        or not re.fullmatch(r"[0-9a-f]{64}", run_nonce)
        or not isinstance(evidence_path, str)
        or not evidence_path.endswith("/sphere/worker-evidence.json")
    ):
        raise ValueError("SPHERE worker run identity is malformed")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("SPHERE worker run id drifted")
    if expected_run_nonce is not None and run_nonce != expected_run_nonce:
        raise ValueError("SPHERE worker run nonce drifted")
    if (
        expected_evidence_path is not None
        and Path(evidence_path) != expected_evidence_path
    ):
        raise ValueError("SPHERE worker evidence path drifted")
    if payload.get("registry", {}).get("sha256") != sha256_file(registry_path):
        raise ValueError("SPHERE registry hash binding drifted")

    runtime = payload.get("runtime", {})
    if runtime.get("world_size") != 1 or runtime.get("rank") != 0:
        raise ValueError("SPHERE worker evidence is not from the frozen one-GPU rank")

    contract = payload.get("contract", {})
    expected_contract = {
        "layer_mode": variant.layer_mode,
        "target_scale_mode": variant.target_scale_mode,
        "fixed_alpha": variant.fixed_alpha,
        "coefficient_mode": variant.coefficient_mode,
        "rho": variant.rho,
        "epsilon": variant.epsilon,
        "feature_site": variant.feature_site,
        "token_selection": variant.token_selection,
        "layer_aggregation": variant.layer_aggregation,
        "physical_microbatch_penalty": False,
        "logical_minibatch_exact_penalty": True,
        "logical_minibatch_surrogate_replay": True,
        "logical_minibatch_gradient_ratio": True,
        "optimizer_gradient_assembly": "explicit_ppo_plus_lambda_sphere",
        "gradient_checkpointing_required_off": True,
        "torch_compile_required_off": True,
    }
    if contract != expected_contract:
        raise ValueError(f"SPHERE worker contract drifted: {contract!r}")

    targets = payload.get("targets", {})
    names = targets.get("names")
    if not isinstance(names, list) or targets.get("count") != len(names):
        raise ValueError("SPHERE target inventory is malformed")
    indices = _canonical_layer_indices(names)
    expected_indices = (
        [variant.expected_hidden_layers - 1]
        if variant.layer_mode == "last"
        else list(range(variant.expected_hidden_layers))
    )
    if indices != expected_indices:
        raise ValueError(
            f"SPHERE target inventory drifted: {indices} != {expected_indices}"
        )

    passes = payload.get("passes", {})
    expected_pass_counts = {
        "ppo_only_forward_backward": 1,
        "sphere_only_forward_backward": 1,
        "combined_forward_backward": 0,
        "explicit_gradient_combination": 1,
        "optimizer_steps": 1,
    }
    expected_pass_keys = set(expected_pass_counts) | {
        "collection",
        "logical_objective",
        "sphere",
    }
    if set(passes) != expected_pass_keys:
        raise ValueError(
            "SPHERE pass inventory drifted: "
            f"actual={sorted(passes)}, expected={sorted(expected_pass_keys)}"
        )
    for field, expected in expected_pass_counts.items():
        if passes.get(field) != expected:
            raise ValueError(
                f"SPHERE pass count drifted for {field}: "
                f"{passes.get(field)!r} != {expected}"
            )

    collection = passes.get("collection", {})
    logical = passes.get("logical_objective", {})
    microbatches = int(logical.get("microbatches", 0))
    if microbatches <= 0 or collection.get("microbatches") != microbatches:
        raise ValueError("SPHERE logical/collection microbatch count drifted")
    token_rows_per_microbatch = logical.get("token_rows_per_microbatch")
    if (
        not isinstance(token_rows_per_microbatch, list)
        or len(token_rows_per_microbatch) != microbatches
        or any(int(value) <= 0 for value in token_rows_per_microbatch)
    ):
        raise ValueError("SPHERE logical token-row partition is malformed")
    total_rows = sum(int(value) for value in token_rows_per_microbatch)
    logical_penalty = _finite_positive(
        logical.get("penalty_sum"), "logical-minibatch SPHERE penalty"
    )
    logical_surrogate_dot = _finite(
        logical.get("surrogate_dot_sum"), "logical surrogate dot sum"
    )
    _finite(logical.get("surrogate_correction"), "logical surrogate correction")

    collection_layers = collection.get("layers", {})
    logical_layers = logical.get("layers", {})
    if list(collection_layers) != names or list(logical_layers) != names:
        raise ValueError("SPHERE logical target layer order drifted")
    layer_penalty_sum = 0.0
    for name in names:
        collected = collection_layers[name]
        layer = logical_layers[name]
        if int(collected.get("calls", 0)) != microbatches:
            raise ValueError(f"SPHERE collection call count drifted at {name}")
        if int(collected.get("token_rows", 0)) != total_rows:
            raise ValueError(f"SPHERE collection token rows drifted at {name}")
        if int(collected.get("feature_dim", 0)) != 9216:
            raise ValueError(f"SPHERE collection feature dimension drifted at {name}")
        if int(layer.get("calls", 0)) != microbatches:
            raise ValueError(f"SPHERE logical hook call count drifted at {name}")
        if int(layer.get("token_rows", 0)) != total_rows:
            raise ValueError(f"SPHERE logical token rows drifted at {name}")
        if int(layer.get("feature_dim", 0)) != 9216:
            raise ValueError(f"SPHERE logical feature dimension drifted at {name}")
        layer_penalty_sum += _finite_positive(
            layer.get("penalty"), f"logical SPHERE penalty at {name}"
        )
        alpha = _finite_positive(layer.get("alpha"), f"SPHERE alpha at {name}")
        if variant.fixed_alpha is not None:
            _require_close(
                alpha,
                variant.fixed_alpha,
                f"fixed alpha at {name}",
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        _finite_positive(
            layer.get("feature_gradient_l2_norm"),
            f"logical feature-gradient norm at {name}",
        )
        _finite(layer.get("surrogate_dot"), f"surrogate dot at {name}")
    _require_close(
        layer_penalty_sum,
        logical_penalty,
        "layerwise logical SPHERE penalty sum",
        rel_tol=5e-6,
        abs_tol=1e-2,
    )

    phase_name = "sphere"
    phase = passes.get(phase_name, {})
    if (
        int(phase.get("microbatches", 0)) != microbatches
        or int(phase.get("identity_matches", 0)) != microbatches
    ):
        raise ValueError("sphere replay coverage drifted")
    _require_close(
        phase.get("penalty_sum"),
        logical_penalty,
        "sphere logical penalty",
        rel_tol=5e-6,
        abs_tol=1e-2,
    )
    surrogate_sum = _require_close(
        phase.get("surrogate_sum"),
        logical_penalty,
        "sphere replay surrogate",
        rel_tol=5e-4,
        abs_tol=1e-2,
    )
    if phase.get("surrogate_matches_penalty") is not True:
        raise ValueError("sphere replay surrogate self-check failed")
    _require_close(
        phase.get("collection_surrogate_dot_sum"),
        logical_surrogate_dot,
        "sphere collection surrogate dot",
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
    raw_surrogate_sum = _finite(
        phase.get("raw_surrogate_sum"), "sphere replay raw surrogate"
    )
    raw_difference = raw_surrogate_sum - logical_surrogate_dot
    _require_close(
        phase.get("raw_surrogate_difference"),
        raw_difference,
        "sphere replay raw reduction difference",
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
    raw_relative_difference = abs(raw_difference) / max(abs(logical_surrogate_dot), 1.0)
    _require_close(
        phase.get("raw_surrogate_relative_difference"),
        raw_relative_difference,
        "sphere replay raw relative reduction difference",
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    _require_close(
        phase.get("raw_surrogate_relative_tolerance"),
        SPHERE_REPLAY_SURROGATE_RELATIVE_TOLERANCE,
        "sphere replay raw surrogate tolerance",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    if (
        phase.get("raw_surrogate_within_tolerance") is not True
        or raw_relative_difference > SPHERE_REPLAY_SURROGATE_RELATIVE_TOLERANCE
    ):
        raise ValueError("sphere replay raw surrogate drift exceeds tolerance")

    layers = phase.get("layers", {})
    if list(layers) != names:
        raise ValueError("sphere replay target order drifted")
    aggregate_difference_squared = 0.0
    aggregate_reference_squared = 0.0
    aggregate_max_difference = 0.0
    aggregate_reference_max = 0.0
    aggregate_different_values = 0
    aggregate_total_values = 0
    for name, layer in layers.items():
        if int(layer.get("calls", 0)) != microbatches:
            raise ValueError(f"sphere replay call count drifted at {name}")
        if int(layer.get("token_rows", 0)) != total_rows:
            raise ValueError(f"sphere replay token rows drifted at {name}")
        feature_dim = int(layer.get("feature_dim", 0))
        if feature_dim != 9216:
            raise ValueError(f"sphere replay feature dimension drifted at {name}")
        _finite(
            layer.get("surrogate_sum_before_global_correction"),
            f"sphere replay surrogate at {name}",
        )
        difference_l2 = _finite_nonnegative(
            layer.get("feature_difference_l2_norm"),
            f"sphere replay feature difference at {name}",
        )
        reference_l2 = _finite_positive(
            layer.get("feature_reference_l2_norm"),
            f"sphere replay feature reference norm at {name}",
        )
        relative_l2 = _finite_nonnegative(
            layer.get("feature_relative_l2_difference"),
            f"sphere replay relative feature difference at {name}",
        )
        _require_close(
            relative_l2,
            difference_l2 / reference_l2,
            f"sphere replay relative L2 identity at {name}",
            rel_tol=1e-8,
            abs_tol=1e-12,
        )
        max_difference = _finite_nonnegative(
            layer.get("feature_max_abs_difference"),
            f"sphere replay max feature difference at {name}",
        )
        reference_max = _finite_positive(
            layer.get("feature_reference_max_abs"),
            f"sphere replay feature reference max at {name}",
        )
        relative_max = _finite_nonnegative(
            layer.get("feature_relative_max_difference"),
            f"sphere replay relative max difference at {name}",
        )
        _require_close(
            relative_max,
            max_difference / max(reference_max, 1e-12),
            f"sphere replay relative max identity at {name}",
            rel_tol=1e-8,
            abs_tol=1e-12,
        )
        total_values = int(layer.get("feature_total_values", 0))
        different_values = int(layer.get("feature_different_values", -1))
        if total_values != total_rows * feature_dim:
            raise ValueError(f"sphere replay feature value count drifted at {name}")
        if not 0 <= different_values <= total_values:
            raise ValueError(f"sphere replay different-value count drifted at {name}")
        layer_within_tolerance = (
            relative_l2 <= SPHERE_REPLAY_RELATIVE_L2_TOLERANCE
            and relative_max <= SPHERE_REPLAY_RELATIVE_MAX_TOLERANCE
        )
        if (
            layer.get("feature_replay_within_tolerance") is not True
            or not layer_within_tolerance
        ):
            raise ValueError(f"sphere replay feature drift exceeds tolerance at {name}")
        aggregate_difference_squared += difference_l2 * difference_l2
        aggregate_reference_squared += reference_l2 * reference_l2
        aggregate_max_difference = max(aggregate_max_difference, max_difference)
        aggregate_reference_max = max(aggregate_reference_max, reference_max)
        aggregate_different_values += different_values
        aggregate_total_values += total_values

    feature_replay = phase.get("feature_replay", {})
    difference_l2 = _finite_nonnegative(
        feature_replay.get("difference_l2_norm"),
        "sphere replay aggregate feature difference",
    )
    reference_l2 = _finite_positive(
        feature_replay.get("reference_l2_norm"),
        "sphere replay aggregate reference norm",
    )
    relative_l2 = _finite_nonnegative(
        feature_replay.get("relative_l2_difference"),
        "sphere replay aggregate relative L2 difference",
    )
    max_difference = _finite_nonnegative(
        feature_replay.get("max_abs_difference"),
        "sphere replay aggregate max difference",
    )
    reference_max = _finite_positive(
        feature_replay.get("reference_max_abs"),
        "sphere replay aggregate reference max",
    )
    relative_max = _finite_nonnegative(
        feature_replay.get("relative_max_difference"),
        "sphere replay aggregate relative max difference",
    )
    expected_difference_l2 = math.sqrt(aggregate_difference_squared)
    expected_reference_l2 = math.sqrt(aggregate_reference_squared)
    for actual, expected, label in (
        (difference_l2, expected_difference_l2, "aggregate difference L2"),
        (reference_l2, expected_reference_l2, "aggregate reference L2"),
        (relative_l2, difference_l2 / reference_l2, "aggregate relative L2"),
        (max_difference, aggregate_max_difference, "aggregate max difference"),
        (reference_max, aggregate_reference_max, "aggregate reference max"),
        (
            relative_max,
            max_difference / max(reference_max, 1e-12),
            "aggregate relative max",
        ),
    ):
        _require_close(
            actual,
            expected,
            f"sphere replay {label}",
            rel_tol=1e-8,
            abs_tol=1e-10,
        )
    if int(feature_replay.get("different_values", -1)) != aggregate_different_values:
        raise ValueError("sphere replay aggregate different-value count drifted")
    if int(feature_replay.get("total_values", 0)) != aggregate_total_values:
        raise ValueError("sphere replay aggregate total-value count drifted")
    _require_close(
        feature_replay.get("relative_l2_tolerance"),
        SPHERE_REPLAY_RELATIVE_L2_TOLERANCE,
        "sphere replay relative L2 tolerance",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    _require_close(
        feature_replay.get("relative_max_tolerance"),
        SPHERE_REPLAY_RELATIVE_MAX_TOLERANCE,
        "sphere replay relative max tolerance",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    if (
        feature_replay.get("within_tolerance") is not True
        or relative_l2 > SPHERE_REPLAY_RELATIVE_L2_TOLERANCE
        or relative_max > SPHERE_REPLAY_RELATIVE_MAX_TOLERANCE
    ):
        raise ValueError("sphere replay aggregate feature drift exceeds tolerance")
    replay_summary = {
        "sphere": {
            "microbatches": microbatches,
            "identity_matches": microbatches,
            "surrogate_sum": surrogate_sum,
            "raw_surrogate_sum": raw_surrogate_sum,
            "raw_surrogate_relative_difference": raw_relative_difference,
            "feature_relative_l2_difference": relative_l2,
            "feature_relative_max_difference": relative_max,
            "feature_different_values": aggregate_different_values,
            "feature_total_values": aggregate_total_values,
        }
    }

    assembly = payload.get("gradient_assembly", {})
    expected_assembly_fields = {
        "mode",
        "parameter_tensors",
        "element_count",
        "finite",
        "accumulation_dtype",
        "ppo_source_dtype_counts",
        "sphere_source_dtype_counts",
        "destination_dtype_counts",
        "ppo_representation_counts",
        "sphere_representation_counts",
        "destination_representation_counts",
        "ppo_placement_counts",
        "sphere_placement_counts",
        "destination_placement_counts",
        "ppo_l2_norm",
        "sphere_l2_norm",
        "ppo_sphere_dot_product",
        "precast_expected_l2_norm",
        "postcast_expected_l2_norm",
        "actual_l2_norm",
        "full_tensor_residual_l2_norm",
        "full_tensor_residual_max_abs",
        "postcast_delta_from_ppo_l2_norm",
        "postcast_delta_from_ppo_max_abs",
        "postcast_values_different_from_ppo",
        "postcast_quantization_error_l2_norm",
        "postcast_quantization_error_max_abs",
    }
    if set(assembly) != expected_assembly_fields:
        raise ValueError("SPHERE optimizer gradient assembly fields drifted")
    if assembly.get("mode") != "explicit_ppo_plus_lambda_sphere":
        raise ValueError("SPHERE optimizer gradient assembly mode drifted")
    if assembly.get("finite") is not True:
        raise ValueError("SPHERE optimizer gradient assembly is non-finite")
    if assembly.get("accumulation_dtype") != "torch.float32":
        raise ValueError("SPHERE optimizer gradient accumulation dtype drifted")
    assembly_tensors = int(assembly.get("parameter_tensors", 0))
    element_count = int(assembly.get("element_count", 0))
    allowed_dtypes = {"torch.float32", "torch.bfloat16", "torch.float16"}
    count_fields = (
        "ppo_source_dtype_counts",
        "sphere_source_dtype_counts",
        "destination_dtype_counts",
        "ppo_representation_counts",
        "sphere_representation_counts",
        "destination_representation_counts",
        "ppo_placement_counts",
        "sphere_placement_counts",
        "destination_placement_counts",
    )
    for field in count_fields:
        counts = assembly.get(field)
        if (
            not isinstance(counts, dict)
            or not counts
            or any(not isinstance(key, str) or not key for key in counts)
            or any(int(count) <= 0 for count in counts.values())
            or sum(int(count) for count in counts.values()) != assembly_tensors
        ):
            raise ValueError(
                f"SPHERE optimizer gradient assembly inventory is malformed: {field}"
            )
    for field in (
        "ppo_source_dtype_counts",
        "sphere_source_dtype_counts",
        "destination_dtype_counts",
    ):
        if not set(assembly[field]) <= allowed_dtypes:
            raise ValueError(
                f"SPHERE optimizer gradient assembly dtype is unsupported: {field}"
            )
    if assembly_tensors <= 0 or element_count <= 0:
        raise ValueError("SPHERE optimizer gradient assembly inventory is malformed")

    gradients = payload.get("gradients", {})
    expected_gradient_keys = {
        "ppo",
        "sphere",
        "optimizer_assembled",
        "lambda",
        "expected_lambda",
        "regularizer_contribution_l2_norm",
        "regularizer_to_ppo_norm_ratio",
        "expected_regularizer_to_ppo_norm_ratio",
        "predicted_optimizer_projection",
        "actual_optimizer_projection",
        "optimizer_projection_residual",
        "optimizer_projection_tolerance",
        "optimizer_reported_preclip_grad_norm",
        "target_activation_gradients",
    }
    if set(gradients) != expected_gradient_keys:
        raise ValueError("SPHERE gradient evidence fields drifted")
    ppo_norm, ppo_projection = _validate_gradient_row(gradients.get("ppo", {}), "PPO")
    sphere_norm, sphere_projection = _validate_gradient_row(
        gradients.get("sphere", {}), "SPHERE"
    )
    assembled_norm, assembled_projection = _validate_gradient_row(
        gradients.get("optimizer_assembled", {}), "assembled optimizer"
    )
    if int(gradients["optimizer_assembled"]["tensors_with_grad"]) != assembly_tensors:
        raise ValueError("SPHERE assembled gradient tensor inventory drifted")
    tensor_counts = {
        int(gradients[name]["tensors_with_grad"])
        for name in ("ppo", "sphere", "optimizer_assembled")
    }
    projection_counts = {
        int(gradients[name]["projection_sample_count"])
        for name in ("ppo", "sphere", "optimizer_assembled")
    }
    if tensor_counts != {assembly_tensors} or len(projection_counts) != 1:
        raise ValueError("SPHERE PPO/SPHERE/assembled gradient coverage drifted")
    coefficient = _finite_positive(gradients.get("lambda"), "SPHERE lambda")
    expected_lambda = variant.rho * ppo_norm / (sphere_norm + variant.epsilon)
    _require_close(
        coefficient,
        expected_lambda,
        "SPHERE lambda formula",
        rel_tol=1e-9,
        abs_tol=0.0,
    )
    _require_close(
        gradients.get("expected_lambda"),
        expected_lambda,
        "reported expected SPHERE lambda",
        rel_tol=1e-9,
        abs_tol=0.0,
    )
    _require_close(
        gradients.get("regularizer_contribution_l2_norm"),
        coefficient * sphere_norm,
        "SPHERE regularizer contribution norm",
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    ratio = _finite_positive(
        gradients.get("regularizer_to_ppo_norm_ratio"),
        "SPHERE contribution ratio",
    )
    expected_ratio = coefficient * sphere_norm / ppo_norm
    _require_close(
        ratio,
        expected_ratio,
        "SPHERE contribution ratio formula",
        rel_tol=1e-9,
        abs_tol=1e-15,
    )
    _require_close(
        gradients.get("expected_regularizer_to_ppo_norm_ratio"),
        expected_ratio,
        "reported expected SPHERE ratio",
        rel_tol=1e-9,
        abs_tol=1e-15,
    )

    assembly_ppo_norm = _require_close(
        assembly.get("ppo_l2_norm"),
        ppo_norm,
        "full-tensor assembly PPO norm",
        rel_tol=1e-5,
        abs_tol=1e-7,
    )
    assembly_sphere_norm = _require_close(
        assembly.get("sphere_l2_norm"),
        sphere_norm,
        "full-tensor assembly SPHERE norm",
        rel_tol=1e-5,
        abs_tol=1e-7,
    )
    ppo_sphere_dot = _finite(
        assembly.get("ppo_sphere_dot_product"),
        "full-tensor PPO/SPHERE dot product",
    )
    cauchy_bound = assembly_ppo_norm * assembly_sphere_norm
    if abs(ppo_sphere_dot) > cauchy_bound * (1.0 + 1e-5) + 1e-7:
        raise ValueError("SPHERE full-tensor dot product violates Cauchy bound")
    expected_precast_squared = (
        assembly_ppo_norm * assembly_ppo_norm
        + 2.0 * coefficient * ppo_sphere_dot
        + coefficient * coefficient * assembly_sphere_norm * assembly_sphere_norm
    )
    if expected_precast_squared < -1e-6:
        raise ValueError("SPHERE expected assembled gradient norm squared is negative")
    expected_precast_norm = math.sqrt(max(expected_precast_squared, 0.0))
    _require_close(
        assembly.get("precast_expected_l2_norm"),
        expected_precast_norm,
        "full-tensor precast expected gradient norm",
        rel_tol=2e-5,
        abs_tol=1e-7,
    )
    postcast_expected_norm = _finite_positive(
        assembly.get("postcast_expected_l2_norm"),
        "full-tensor postcast expected gradient norm",
    )
    actual_full_norm = _finite_positive(
        assembly.get("actual_l2_norm"),
        "full-tensor actual optimizer gradient norm",
    )
    _require_close(
        actual_full_norm,
        postcast_expected_norm,
        "full-tensor postcast expected/actual gradient norm",
        rel_tol=1e-7,
        abs_tol=1e-8,
    )
    _require_close(
        actual_full_norm,
        assembled_norm,
        "full-tensor actual/reported assembled gradient norm",
        rel_tol=1e-5,
        abs_tol=1e-7,
    )
    residual_l2 = _finite_nonnegative(
        assembly.get("full_tensor_residual_l2_norm"),
        "full-tensor assembly residual L2 norm",
    )
    residual_max = _finite_nonnegative(
        assembly.get("full_tensor_residual_max_abs"),
        "full-tensor assembly residual max",
    )
    if residual_l2 != 0.0 or residual_max != 0.0:
        raise ValueError("SPHERE full-tensor optimizer assembly residual is nonzero")
    postcast_delta_l2 = _finite_positive(
        assembly.get("postcast_delta_from_ppo_l2_norm"),
        "postcast SPHERE contribution L2 norm",
    )
    postcast_delta_max = _finite_positive(
        assembly.get("postcast_delta_from_ppo_max_abs"),
        "postcast SPHERE contribution max",
    )
    changed_values = int(assembly.get("postcast_values_different_from_ppo", 0))
    if not 0 < changed_values <= element_count:
        raise ValueError("SPHERE contribution did not survive the destination cast")
    _finite_nonnegative(
        assembly.get("postcast_quantization_error_l2_norm"),
        "postcast assembly quantization-error L2 norm",
    )
    _finite_nonnegative(
        assembly.get("postcast_quantization_error_max_abs"),
        "postcast assembly quantization-error max",
    )
    if postcast_delta_l2 <= 0.0 or postcast_delta_max <= 0.0:
        raise ValueError("SPHERE contribution did not survive the destination cast")

    predicted_projection = ppo_projection + coefficient * sphere_projection
    _require_close(
        gradients.get("predicted_optimizer_projection"),
        predicted_projection,
        "predicted optimizer gradient projection",
        rel_tol=1e-9,
        abs_tol=1e-10,
    )
    _require_close(
        gradients.get("actual_optimizer_projection"),
        assembled_projection,
        "actual optimizer gradient projection",
        rel_tol=1e-9,
        abs_tol=1e-10,
    )
    residual = _finite(
        gradients.get("optimizer_projection_residual"),
        "optimizer gradient projection residual",
    )
    _require_close(
        residual,
        assembled_projection - predicted_projection,
        "optimizer projection residual identity",
        rel_tol=1e-9,
        abs_tol=1e-10,
    )
    tolerance = _finite_positive(
        gradients.get("optimizer_projection_tolerance"),
        "optimizer projection tolerance",
    )
    expected_tolerance = max(
        1e-5,
        5e-4 * max(abs(predicted_projection), abs(assembled_projection), 1.0),
    )
    _require_close(
        tolerance,
        expected_tolerance,
        "optimizer projection tolerance contract",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    if abs(residual) > tolerance:
        raise ValueError(
            f"optimizer gradient projection residual {residual} exceeds {tolerance}"
        )
    optimizer_norm = _finite_positive(
        gradients.get("optimizer_reported_preclip_grad_norm"),
        "optimizer-reported preclip gradient norm",
    )
    _require_close(
        optimizer_norm,
        assembled_norm,
        "optimizer-reported versus assembled gradient norm",
        rel_tol=1e-5,
        abs_tol=1e-7,
    )

    activation = gradients.get("target_activation_gradients", {})
    if list(activation) != names:
        raise ValueError("SPHERE activation-gradient target inventory drifted")
    for name, row in activation.items():
        if (
            row.get("gradient_finite") is not True
            or row.get("gradient_nonzero") is not True
        ):
            raise ValueError(f"SPHERE activation gradient failed at {name}")
        _finite_positive(row.get("gradient_l2_norm"), f"activation gradient at {name}")

    checks = payload.get("checks", {})
    expected_checks = {
        "logical_minibatch_penalty_finite_nonzero",
        "logical_feature_gradients_finite_nonzero",
        "sphere_gradient_finite_nonzero",
        "lambda_formula_matches",
        "ratio_formula_matches",
        "sphere_replay_matches_logical_penalty",
        "sphere_replay_features_within_tolerance",
        "sphere_replay_surrogate_within_tolerance",
        "all_replay_microbatch_identities_match",
        "optimizer_gradient_assembly_finite",
        "full_tensor_assembly_exact",
        "postcast_regularizer_contribution_nonzero",
        "optimizer_projection_matches_explicit_sum",
        "optimizer_preclip_grad_norm_finite_nonzero",
        "all_target_activation_gradients_nonzero",
        "optimizer_step_exactly_one",
    }
    if set(checks) != expected_checks:
        raise ValueError("SPHERE worker check inventory drifted")
    if not all(value is True for value in checks.values()):
        raise ValueError(f"SPHERE worker checks failed: {checks}")
    for check_name in (
        "sphere_replay_features_within_tolerance",
        "sphere_replay_surrogate_within_tolerance",
        "optimizer_gradient_assembly_finite",
        "full_tensor_assembly_exact",
        "postcast_regularizer_contribution_nonzero",
        "optimizer_projection_matches_explicit_sum",
    ):
        if checks.get(check_name) is not True:
            raise ValueError(f"SPHERE worker check is missing/failed: {check_name}")

    return {
        "target_count": len(names),
        "target_layer_indices": indices,
        "logical_minibatch_microbatches": microbatches,
        "logical_minibatch_token_rows": total_rows,
        "logical_minibatch_penalty": logical_penalty,
        "ppo_gradient_l2_norm": ppo_norm,
        "sphere_gradient_l2_norm": sphere_norm,
        "assembled_optimizer_gradient_l2_norm": assembled_norm,
        "full_tensor_assembly_residual_l2_norm": residual_l2,
        "full_tensor_assembly_residual_max_abs": residual_max,
        "postcast_regularizer_delta_l2_norm": postcast_delta_l2,
        "postcast_regularizer_delta_max_abs": postcast_delta_max,
        "postcast_values_different_from_ppo": changed_values,
        "lambda": coefficient,
        "regularizer_to_ppo_norm_ratio": ratio,
        "optimizer_projection_residual": residual,
        "optimizer_projection_tolerance": tolerance,
        "optimizer_reported_preclip_grad_norm": optimizer_norm,
        "replays": replay_summary,
    }


def _yaml_scalar(text: str, dotted_path: str) -> str:
    wanted = dotted_path.split(".")
    stack: list[tuple[int, str]] = []
    values: dict[tuple[str, ...], str] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if stripped.startswith("-") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple([item[1] for item in stack] + [key.strip()])
        value = value.strip()
        if value:
            values[path] = value.strip("'\"")
        else:
            stack.append((indent, key.strip()))
    key = tuple(wanted)
    if key not in values:
        raise ValueError(f"resolved config is missing {dotted_path}")
    return values[key]


def validate_resolved_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    expected = {
        "model_engine": "dp",
        "data.train_batch_size": "400",
        "actor_rollout_ref.model.enable_gradient_checkpointing": "false",
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.use_orig_params": "false",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "400",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "4",
        "actor_rollout_ref.actor.ppo_epochs": "1",
        "actor_rollout_ref.actor.use_torch_compile": "false",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": "false",
        "actor_rollout_ref.rollout.n": "2",
        "trainer.total_training_steps": "1",
        "trainer.save_freq": "1",
        "trainer.resume_mode": "disable",
    }
    actual = {key: _yaml_scalar(text, key) for key in expected}
    if actual != expected:
        raise ValueError(f"resolved veRL configuration drifted: {actual!r}")
    return actual


def validate_rollout_ledger(
    path: Path,
    *,
    expected_trajectories: int,
    expected_groups: int,
    rollouts_per_group: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_trajectories:
        raise ValueError(
            f"trajectory count drifted: {len(rows)} != {expected_trajectories}"
        )
    if {int(row["step"]) for row in rows} != {1}:
        raise ValueError("rollout ledger contains a non-one-step global step")
    groups: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        score = _finite(row["score"], "rollout score")
        groups[str(row["input"])].append(score)
    if len(groups) != expected_groups:
        raise ValueError(
            f"GRPO group count drifted: {len(groups)} != {expected_groups}"
        )
    if any(len(scores) != rollouts_per_group for scores in groups.values()):
        raise ValueError("GRPO group size drifted")
    mixed = sum(min(scores) != max(scores) for scores in groups.values())
    if mixed <= 0:
        raise ValueError("no mixed-reward GRPO group; PPO gradient is not demonstrated")
    return {
        "trajectory_count": len(rows),
        "grpo_group_count": len(groups),
        "rollouts_per_group": rollouts_per_group,
        "mixed_reward_group_count": mixed,
        "reward_values": sorted({float(row["score"]) for row in rows}),
    }


def build_checkpoint_inventory(run_dir: Path) -> dict[str, Any]:
    checkpoint_root = run_dir / "checkpoints"
    step_dirs = sorted(
        (
            (int(match.group(1)), path)
            for path in checkpoint_root.glob("global_step_*")
            if path.is_dir() and (match := _STEP_RE.fullmatch(path.name))
        ),
        key=lambda item: item[0],
    )
    if [step for step, _ in step_dirs] != [1]:
        raise ValueError(
            f"expected exactly global_step_1 checkpoint, got {[step for step, _ in step_dirs]}"
        )
    checkpoint = step_dirs[0][1]
    required = [
        checkpoint / "data.pt",
        checkpoint / "actor" / "huggingface" / "config.json",
    ]
    if any(not path.is_file() for path in required):
        raise ValueError("global_step_1 checkpoint is incomplete")
    if not list((checkpoint / "actor" / "huggingface").glob("*.safetensors")):
        raise ValueError("global_step_1 has no Hugging Face safetensors")

    files = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError(f"checkpoint contains a symlink: {path}")
        files.append(
            {
                "path": path.relative_to(checkpoint_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("global_step_1 checkpoint contains no files")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": run_dir.name,
        "global_step": 1,
        "files": files,
    }


def _validate_parameter_evidence(
    payload: dict[str, Any], *, run_dir: Path, checkpoint_inventory: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("schema") != PARAMETER_SCHEMA:
        raise ValueError("unexpected parameter evidence schema")
    if payload.get("weights_changed") is not True:
        raise ValueError("parameter audit did not demonstrate changed weights")
    if payload.get("effective_bfloat16_weights_changed") is not True:
        raise ValueError("parameter audit did not demonstrate an effective BF16 change")
    if int(payload.get("changed_probe_count", 0)) <= 0:
        raise ValueError("parameter audit has no changed probes")
    after = run_dir / "checkpoints/global_step_1/actor/huggingface"
    if Path(payload.get("after_path", "")).resolve() != after.resolve():
        raise ValueError("parameter evidence after_path is not global_step_1")
    if payload.get("after_config_sha256") != sha256_file(after / "config.json"):
        raise ValueError("parameter evidence after config hash drifted")

    inventory_by_name = {
        row["path"]: row for row in checkpoint_inventory.get("files", [])
    }
    for row in payload.get("evaluated_weight_files", {}).get("after", []):
        relative = f"global_step_1/actor/huggingface/{row['name']}"
        if relative not in inventory_by_name:
            raise ValueError(
                f"parameter-evaluated weight is absent from checkpoint: {relative}"
            )
        if row.get("sha256") != inventory_by_name[relative]["sha256"]:
            raise ValueError(f"parameter/checkpoint weight hash drifted: {relative}")
    return {
        "weights_changed": True,
        "effective_bfloat16_weights_changed": True,
        "changed_probe_count": int(payload["changed_probe_count"]),
    }


def _validate_run_contract(
    payload: dict[str, Any],
    *,
    run_dir: Path,
    expected_variant: str,
    expected_run_nonce_sha256: str,
) -> dict[str, Any]:
    if payload.get("schema") != RUN_CONTRACT_SCHEMA:
        raise ValueError("unexpected SPHERE run-contract schema")
    if (
        payload.get("run_id") != run_dir.name
        or payload.get("variant") != expected_variant
    ):
        raise ValueError("SPHERE run-contract identity drifted")
    algorithm = payload.get("algorithm", {})
    expected_algorithm = {
        "base": "GRPO",
        "sphere_variant": expected_variant,
        "rho": 1e-3,
        "rollout_n": 2,
        "train_batch_size": 400,
        "trajectory_count": 800,
        "ppo_epochs": 1,
        "ppo_micro_batch_size_per_gpu": 4,
        "actor_lr": 1e-6,
        "thinking": False,
        "gradient_checkpointing": False,
        "torch_compile": False,
        "logical_minibatch_exact_penalty": True,
        "backward_passes": {"ppo": 1, "sphere": 1, "combined": 0},
        "optimizer_gradient_assembly": "explicit_ppo_plus_lambda_sphere",
        "full_tensor_postcast_assembly_check": True,
    }
    if algorithm != expected_algorithm:
        raise ValueError(f"SPHERE run-contract algorithm drifted: {algorithm!r}")
    if payload.get("run_nonce_sha256") != expected_run_nonce_sha256:
        raise ValueError("SPHERE run-contract nonce binding drifted")
    if payload.get("runtime", {}).get("verl_commit") != EXPECTED_VERL_COMMIT:
        raise ValueError("SPHERE run-contract veRL commit drifted")
    constraints = payload.get("constraints", {})
    for key in (
        "no_intermediate_evaluation",
        "restore_holder_after_run",
        "do_not_release_allocation",
        "no_new_allocation",
    ):
        if constraints.get(key) is not True:
            raise ValueError(f"SPHERE run-contract constraint is not true: {key}")
    if payload.get("data", {}).get("test_jsonl_sha256") != EXPECTED_TEST_JSONL_SHA256:
        raise ValueError("SPHERE run-contract test dataset hash drifted")
    return expected_algorithm


def audit_training_run(
    *,
    run_dir: Path,
    expected_variant: str,
    registry_path: Path,
    expected_trajectories: int = 800,
    expected_groups: int = 400,
    rollouts_per_group: int = 2,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Audit a completed one-update run; return evidence, inventory, receipt."""
    worker_path = run_dir / "sphere/worker-evidence.json"
    failed_worker_path = run_dir / "sphere/worker-evidence.failed.json"
    nonce_path = run_dir / "sphere/run-nonce"
    parameter_path = run_dir / "parameter-evidence.json"
    contract_path = run_dir / "contract/run-contract.json"
    config_path = run_dir / "contract/resolved-config.yaml"
    if failed_worker_path.exists():
        raise ValueError("coexisting failed SPHERE worker evidence exists")
    if worker_path.is_symlink() or nonce_path.is_symlink():
        raise ValueError("SPHERE worker evidence identity files cannot be symlinks")
    run_nonce = nonce_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", run_nonce):
        raise ValueError("SPHERE run nonce is malformed")
    if worker_path.stat().st_mtime_ns < nonce_path.stat().st_mtime_ns:
        raise ValueError("SPHERE worker evidence predates its run nonce")
    run_nonce_sha256 = sha256_file(nonce_path)
    worker_summary = validate_worker_evidence(
        _read_json(worker_path),
        expected_variant=expected_variant,
        registry_path=registry_path,
        expected_run_id=run_dir.name,
        expected_run_nonce=run_nonce,
        expected_evidence_path=worker_path,
    )
    contract_summary = _validate_run_contract(
        _read_json(contract_path),
        run_dir=run_dir,
        expected_variant=expected_variant,
        expected_run_nonce_sha256=run_nonce_sha256,
    )
    config_summary = validate_resolved_config(config_path)

    rollout_files = sorted((run_dir / "rollouts").glob("*.jsonl"))
    if len(rollout_files) != 1:
        raise ValueError(f"expected one rollout ledger, got {rollout_files}")
    rollout_summary = validate_rollout_ledger(
        rollout_files[0],
        expected_trajectories=expected_trajectories,
        expected_groups=expected_groups,
        rollouts_per_group=rollouts_per_group,
    )

    checkpoint_inventory = build_checkpoint_inventory(run_dir)
    parameter_summary = _validate_parameter_evidence(
        _read_json(parameter_path),
        run_dir=run_dir,
        checkpoint_inventory=checkpoint_inventory,
    )

    # The inventory is written before its digest is bound into the two receipts.
    inventory_path = run_dir / "checkpoint-inventory.json"
    _write_json_atomic(inventory_path, checkpoint_inventory)
    artifact_hashes = {
        "worker_evidence_sha256": sha256_file(worker_path),
        "run_nonce_sha256": run_nonce_sha256,
        "rollout_ledger_sha256": sha256_file(rollout_files[0]),
        "sphere_registry_sha256": sha256_file(registry_path),
        "parameter_evidence_sha256": sha256_file(parameter_path),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "run_contract_sha256": sha256_file(contract_path),
        "resolved_config_sha256": sha256_file(config_path),
    }
    evidence = {
        "schema": TRAINING_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "variant": expected_variant,
        "global_step": checkpoint_inventory["global_step"],
        "optimizer_updates_actual": _read_json(worker_path)["passes"][
            "optimizer_steps"
        ],
        **rollout_summary,
        "worker_evidence": worker_summary,
        "parameter_evidence": parameter_summary,
        "resolved_config": config_summary,
        "run_contract_algorithm": contract_summary,
        "artifacts": artifact_hashes,
    }
    training_path = run_dir / "training-evidence.json"
    _write_json_atomic(training_path, evidence)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "variant": expected_variant,
        "global_step": checkpoint_inventory["global_step"],
        "optimizer_updates_actual": evidence["optimizer_updates_actual"],
        "weights_changed": parameter_summary["weights_changed"],
        "effective_bfloat16_weights_changed": parameter_summary[
            "effective_bfloat16_weights_changed"
        ],
        "artifacts": {
            **artifact_hashes,
            "training_evidence_sha256": sha256_file(training_path),
        },
    }
    if completion["global_step"] != 1 or completion["optimizer_updates_actual"] != 1:
        raise ValueError("training completion is not exactly one optimizer/global step")
    completion_path = run_dir / "training-complete.json"
    _write_json_atomic(completion_path, completion)
    return evidence, checkpoint_inventory, completion


def _validate_evaluation(payload: dict[str, Any], label: str) -> dict[str, Any]:
    protocol = payload.get("protocol", {})
    if protocol.get("dataset_sha256") != EXPECTED_TEST_JSONL_SHA256:
        raise ValueError(f"{label} evaluation dataset hash drifted")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 100:
        raise ValueError(f"{label} evaluation does not contain 100 tasks")
    ids = [str(row.get("id")) for row in tasks]
    if len(set(ids)) != 100:
        raise ValueError(f"{label} evaluation task ids are not unique")
    successes = sum(row.get("success") is True for row in tasks)
    summary = payload.get("summary", {})
    if (
        int(summary.get("total", -1)) != 100
        or int(summary.get("successes", -1)) != successes
    ):
        raise ValueError(f"{label} evaluation summary drifted")
    rate = _require_close(
        summary.get("success_rate"), successes / 100.0, f"{label} success rate"
    )
    return {"ids": ids, "successes": successes, "total": 100, "success_rate": rate}


def audit_final_run(
    *, run_dir: Path, expected_variant: str, success_threshold: int = 95
) -> dict[str, Any]:
    """Validate post-training test100 and bind it to the training receipt."""
    completion_path = run_dir / "training-complete.json"
    training_path = run_dir / "training-evidence.json"
    worker_path = run_dir / "sphere/worker-evidence.json"
    parameter_path = run_dir / "parameter-evidence.json"
    inventory_path = run_dir / "checkpoint-inventory.json"
    contract_path = run_dir / "contract/run-contract.json"
    config_path = run_dir / "contract/resolved-config.yaml"
    before_path = run_dir / "evaluation/before-test100.json"
    after_path = run_dir / "evaluation/after-test100.json"

    completion = _read_json(completion_path)
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValueError("unexpected SPHERE training-complete schema")
    if (
        completion.get("run_id") != run_dir.name
        or completion.get("variant") != expected_variant
    ):
        raise ValueError("SPHERE training-complete identity drifted")
    if (
        completion.get("global_step") != 1
        or completion.get("optimizer_updates_actual") != 1
    ):
        raise ValueError("SPHERE training-complete step count drifted")
    expected_hashes = {
        "worker_evidence_sha256": sha256_file(worker_path),
        "run_nonce_sha256": sha256_file(run_dir / "sphere/run-nonce"),
        "rollout_ledger_sha256": sha256_file(
            next(iter(sorted((run_dir / "rollouts").glob("*.jsonl"))))
        ),
        "sphere_registry_sha256": completion["artifacts"]["sphere_registry_sha256"],
        "parameter_evidence_sha256": sha256_file(parameter_path),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "run_contract_sha256": sha256_file(contract_path),
        "resolved_config_sha256": sha256_file(config_path),
        "training_evidence_sha256": sha256_file(training_path),
    }
    if completion.get("artifacts") != expected_hashes:
        raise ValueError("SPHERE training-complete artifact binding drifted")
    if after_path.stat().st_mtime_ns < completion_path.stat().st_mtime_ns:
        raise ValueError("after-test100 predates the training completion receipt")

    before = _validate_evaluation(_read_json(before_path), "before")
    after = _validate_evaluation(_read_json(after_path), "after")
    if before["ids"] != after["ids"]:
        raise ValueError("before/after test100 task identities drifted")
    if after["successes"] < success_threshold:
        raise ValueError(
            f"test100 threshold failed: {after['successes']}/{after['total']}"
        )
    training = _read_json(training_path)
    if (
        training.get("schema") != TRAINING_SCHEMA
        or training.get("variant") != expected_variant
    ):
        raise ValueError("SPHERE training evidence identity drifted")

    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "run_id": run_dir.name,
        "variant": expected_variant,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_step": 1,
        "optimizer_updates_actual": 1,
        "before_successes": before["successes"],
        "before_total": before["total"],
        "before_success_rate": before["success_rate"],
        "after_successes": after["successes"],
        "after_total": after["total"],
        "after_success_rate": after["success_rate"],
        "meets_at_least_95_percent": True,
        "success_rate_delta": after["success_rate"] - before["success_rate"],
        "weights_changed": completion["weights_changed"],
        "effective_bfloat16_weights_changed": completion[
            "effective_bfloat16_weights_changed"
        ],
        "mixed_reward_group_count": training["mixed_reward_group_count"],
        "sphere": training["worker_evidence"],
        "artifacts": {
            **expected_hashes,
            "training_complete_sha256": sha256_file(completion_path),
            "before_test100_sha256": sha256_file(before_path),
            "after_test100_sha256": sha256_file(after_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    training = subparsers.add_parser("training")
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--variant", required=True)
    training.add_argument("--registry", type=Path, required=True)
    training.add_argument("--expected-trajectories", type=int, default=800)
    training.add_argument("--expected-groups", type=int, default=400)
    training.add_argument("--rollouts-per-group", type=int, default=2)

    final = subparsers.add_parser("final")
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--variant", required=True)
    final.add_argument("--success-threshold", type=int, default=95)

    args = parser.parse_args()
    if args.stage == "training":
        evidence, inventory, completion = audit_training_run(
            run_dir=args.run_dir,
            expected_variant=args.variant,
            registry_path=args.registry,
            expected_trajectories=args.expected_trajectories,
            expected_groups=args.expected_groups,
            rollouts_per_group=args.rollouts_per_group,
        )
        payload = {
            "checkpoint_inventory": inventory,
            "training_evidence": evidence,
            "training_complete": completion,
        }
    else:
        payload = audit_final_run(
            run_dir=args.run_dir,
            expected_variant=args.variant,
            success_threshold=args.success_threshold,
        )
        _write_json_atomic(args.run_dir / "result-status.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
