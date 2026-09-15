#!/usr/bin/env python3
"""Audit a veRL Hugging Face export against its frozen base checkpoint.

This script compares semantic tensor identities and sampled tensor values rather
than relying on checkpoint filenames. It therefore supports a sharded BF16 base
checkpoint and a single-file FP32 veRL export of the same policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from intent_alignment_toy.evidence import (
    EXPECTED_CONFIG_NORMALIZATION,
    MISSING,
)

DEFAULT_PROBE_LAYERS = (0, 1, 8, 16, 24, 31)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_file_manifest(root: Path, filenames: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": filename,
            "bytes": (root / filename).stat().st_size,
            "sha256": sha256_file(root / filename),
            "shard_id": (
                filename.removeprefix("model.safetensors-").removesuffix(
                    ".safetensors"
                )
                if filename.startswith("model.safetensors-")
                else filename
            ),
        }
        for filename in filenames
    ]


def json_diff(before: Any, after: Any, pointer: str = "") -> dict[str, Any]:
    if type(before) is not type(after):
        return {pointer: {"before": before, "after": after}}
    if isinstance(before, dict):
        result: dict[str, Any] = {}
        for key in sorted(set(before) | set(after)):
            child_pointer = f"{pointer}/{key}"
            if key not in before:
                result[child_pointer] = {"before": MISSING, "after": after[key]}
            elif key not in after:
                result[child_pointer] = {"before": before[key], "after": MISSING}
            else:
                result.update(json_diff(before[key], after[key], child_pointer))
        return result
    if isinstance(before, list):
        return {} if before == after else {pointer: {"before": before, "after": after}}
    return {} if before == after else {pointer: {"before": before, "after": after}}


def tensor_manifest(root: Path) -> tuple[dict[str, str], list[str]]:
    # Heavy imports stay inside the GPU-runtime path so --help and pure unit
    # tests remain usable on the project Mac, which intentionally has no torch.
    from safetensors import safe_open

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        files = sorted(set(index["weight_map"].values()))
    else:
        files = sorted(path.name for path in root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors files found under {root}")

    tensors: dict[str, str] = {}
    for filename in files:
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open is not a Mapping
                if name in tensors:
                    raise ValueError(f"duplicate tensor key {name!r}")
                tensors[name] = filename
    return tensors, files


def is_visual_tensor(name: str) -> bool:
    return name.startswith("model.visual.") or ".visual." in name


def is_policy_core_tensor(name: str) -> bool:
    return not is_visual_tensor(name) and not name.startswith("mtp.")


def audit_parameter_update(
    base: Path,
    after: Path,
    *,
    expected_policy_core_tensors: int = 426,
    probe_layers: tuple[int, ...] = DEFAULT_PROBE_LAYERS,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    before_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    after_config = json.loads((after / "config.json").read_text(encoding="utf-8"))
    actual_normalization = json_diff(before_config, after_config)
    if actual_normalization != EXPECTED_CONFIG_NORMALIZATION:
        raise ValueError(
            "unexpected config normalization: "
            + json.dumps(actual_normalization, ensure_ascii=False, sort_keys=True)
        )

    before_tensors, before_files = tensor_manifest(base)
    after_tensors, after_files = tensor_manifest(after)
    before_core = {name for name in before_tensors if is_policy_core_tensor(name)}
    after_core = {name for name in after_tensors if is_policy_core_tensor(name)}
    if before_core != after_core:
        raise ValueError("policy core tensor key sets differ")
    if len(before_core) != expected_policy_core_tensors:
        raise ValueError(
            "unexpected policy core tensor count: "
            f"expected {expected_policy_core_tensors}, got {len(before_core)}"
        )

    base_only = set(before_tensors) - set(after_tensors)
    after_only = set(after_tensors) - set(before_tensors)
    base_mtp = sorted(name for name in base_only if name.startswith("mtp."))
    unexpected_base_only = sorted(
        name
        for name in base_only
        if not name.startswith("mtp.") and not is_visual_tensor(name)
    )
    unexpected_after_only = sorted(
        name for name in after_only if not is_visual_tensor(name)
    )
    if unexpected_base_only or unexpected_after_only:
        raise ValueError(
            "unexpected non-core tensor mismatch: "
            f"base_only={unexpected_base_only}, after_only={unexpected_after_only}"
        )

    probes = [
        f"model.language_model.layers.{layer}.input_layernorm.weight"
        for layer in probe_layers
    ]
    probe_rows: list[dict[str, Any]] = []
    for name in probes:
        if name not in before_core:
            raise ValueError(f"missing deterministic probe tensor {name!r}")
        with (
            safe_open(
                base / before_tensors[name], framework="pt", device="cpu"
            ) as before_handle,
            safe_open(
                after / after_tensors[name], framework="pt", device="cpu"
            ) as after_handle,
        ):
            before_native = before_handle.get_tensor(name)
            after_native = after_handle.get_tensor(name)
            before_float = before_native.float().contiguous()
            after_float = after_native.float().contiguous()

        if before_float.shape != after_float.shape:
            raise ValueError(f"probe shape drifted for {name!r}")
        absolute_delta = (after_float - before_float).abs()
        changed_float32 = int((absolute_delta > 0).sum())
        effective_bfloat16 = int(
            (after_float.to(torch.bfloat16) != before_native.to(torch.bfloat16)).sum()
        )
        if changed_float32 == 0:
            raise ValueError(f"probe tensor {name!r} did not change")

        probe_rows.append(
            {
                "name": name,
                "shape": list(before_float.shape),
                "base_dtype": str(before_native.dtype),
                "after_dtype": str(after_native.dtype),
                "base_float32_sha256": hashlib.sha256(
                    before_float.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "after_float32_sha256": hashlib.sha256(
                    after_float.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "changed_float32_values": changed_float32,
                "effective_bfloat16_changed_values": effective_bfloat16,
                "max_abs_delta": float(absolute_delta.max()),
                "mean_abs_delta": float(absolute_delta.mean()),
            }
        )

    effective_bfloat16_changes = sum(
        row["effective_bfloat16_changed_values"] for row in probe_rows
    )
    if effective_bfloat16_changes < 1:
        raise ValueError("no effective BF16 change in deterministic probes")

    return {
        "schema": "intent_alignment_semantic_parameter_evidence_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_path": str(base),
        "after_path": str(after),
        "base_config_sha256": sha256_file(base / "config.json"),
        "after_config_sha256": sha256_file(after / "config.json"),
        "allowed_config_normalization": actual_normalization,
        "policy_core_tensor_count": len(before_core),
        "policy_core_key_sets_equal": True,
        "excluded_base_mtp_tensor_count": len(base_mtp),
        "vision_tensor_count": {
            "before": sum(is_visual_tensor(name) for name in before_tensors),
            "after": sum(is_visual_tensor(name) for name in after_tensors),
        },
        "serialization": {
            "before_files": before_files,
            "after_files": after_files,
        },
        "evaluated_weight_files": {
            "before": weight_file_manifest(base, before_files),
            "after": weight_file_manifest(after, after_files),
        },
        "probe_rule": (
            "input_layernorm weights at layers "
            + ",".join(str(layer) for layer in probe_layers)
        ),
        "probes": probe_rows,
        "changed_probe_count": len(probe_rows),
        "effective_bfloat16_changed_value_count": effective_bfloat16_changes,
        "weights_changed": True,
        "effective_bfloat16_weights_changed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--trained-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-policy-core-tensors", type=int, default=426)
    args = parser.parse_args()

    evidence = audit_parameter_update(
        args.base_model,
        args.trained_model,
        expected_policy_core_tensors=args.expected_policy_core_tensors,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
