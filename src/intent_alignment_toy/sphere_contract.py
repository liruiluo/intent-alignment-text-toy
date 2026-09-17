"""Pure-Python contracts and reference math for the dense SPHERE variants."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LAYER_PATTERN = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj$")
_FSDP_WRAPPER_SEGMENT = "_fsdp_wrapped_module"


@dataclass(frozen=True)
class SphereVariant:
    """Frozen algorithm choices for one dense SPHERE experiment arm."""

    name: str
    layer_mode: str
    target_scale_mode: str
    fixed_alpha: float | None
    coefficient_mode: str
    rho: float
    epsilon: float
    feature_site: str
    token_selection: str
    layer_aggregation: str
    expected_hidden_layers: int

    def validate(self) -> None:
        if self.layer_mode not in {"last", "all_hidden"}:
            raise ValueError(f"unsupported layer_mode: {self.layer_mode}")
        if self.target_scale_mode not in {"adaptive_trace", "fixed"}:
            raise ValueError(f"unsupported target_scale_mode: {self.target_scale_mode}")
        if self.target_scale_mode == "fixed":
            if self.fixed_alpha is None or not math.isfinite(self.fixed_alpha):
                raise ValueError("fixed target scale requires a finite fixed_alpha")
        elif self.fixed_alpha is not None:
            raise ValueError("adaptive target scale must not set fixed_alpha")
        if self.coefficient_mode != "gradient_norm_ratio":
            raise ValueError(f"unsupported coefficient_mode: {self.coefficient_mode}")
        if not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if self.feature_site != "mlp.down_proj.input":
            raise ValueError(f"unsupported feature_site: {self.feature_site}")
        if self.token_selection != "shifted_ppo_loss_mask":
            raise ValueError(f"unsupported token_selection: {self.token_selection}")
        if self.layer_aggregation != "sum":
            raise ValueError(f"unsupported layer_aggregation: {self.layer_aggregation}")
        if self.expected_hidden_layers <= 0:
            raise ValueError("expected_hidden_layers must be positive")


def load_sphere_variant(path: str | Path, name: str) -> SphereVariant:
    """Load and validate one variant from the checked-in JSON registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "intent_alignment_sphere_variants_v1":
        raise ValueError("unexpected SPHERE variant registry schema")
    try:
        specific = payload["variants"][name]
    except KeyError as error:
        raise ValueError(f"unknown SPHERE variant: {name}") from error
    merged = {**payload["common"], **specific}
    variant = SphereVariant(name=name, **merged)
    variant.validate()
    return variant


def sphere_penalty_from_gram_stats(
    *,
    frobenius_squared: Any,
    trace: Any,
    feature_dim: int,
    fixed_alpha: float | None,
) -> tuple[Any, Any]:
    """Compute SPHERE from Gram sufficient statistics.

    ``frobenius_squared`` and ``trace`` may be floats or scalar tensors.  The
    feature dimension is always the column dimension D of Phi, even when the
    implementation uses the smaller sample-space Gram Phi Phi^T.
    """
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    alpha = trace / feature_dim if fixed_alpha is None else fixed_alpha
    penalty = frobenius_squared - 2.0 * alpha * trace + feature_dim * alpha * alpha
    return penalty, alpha


def sphere_penalty_reference(
    rows: Sequence[Sequence[float]], *, fixed_alpha: float | None = None
) -> tuple[float, float]:
    """Small dependency-free reference implementation used by CPU tests."""
    if not rows or not rows[0]:
        raise ValueError("features must be a non-empty rectangular matrix")
    token_count = len(rows)
    feature_dim = len(rows[0])
    if any(len(row) != feature_dim for row in rows):
        raise ValueError("features must be rectangular")

    gram = [
        [
            sum(rows[t][i] * rows[t][j] for t in range(token_count)) / token_count
            for j in range(feature_dim)
        ]
        for i in range(feature_dim)
    ]
    frobenius_squared = sum(value * value for row in gram for value in row)
    trace = sum(gram[index][index] for index in range(feature_dim))
    penalty, alpha = sphere_penalty_from_gram_stats(
        frobenius_squared=frobenius_squared,
        trace=trace,
        feature_dim=feature_dim,
        fixed_alpha=fixed_alpha,
    )
    return float(penalty), float(alpha)


def sphere_penalty_and_feature_gradient_reference(
    rows: Sequence[Sequence[float]], *, fixed_alpha: float | None = None
) -> tuple[float, float, list[list[float]]]:
    """Return the exact logical-batch SPHERE loss and ``dL/dPhi``.

    This dependency-free implementation is deliberately small and slow.  It is
    the executable specification used to check the GPU implementation and, in
    particular, to prevent a nonlinear SPHERE loss from being evaluated
    independently on physical gradient-accumulation microbatches.
    """
    penalty, alpha = sphere_penalty_reference(rows, fixed_alpha=fixed_alpha)
    token_count = len(rows)
    feature_dim = len(rows[0])
    gram = [
        [
            sum(rows[t][i] * rows[t][j] for t in range(token_count)) / token_count
            for j in range(feature_dim)
        ]
        for i in range(feature_dim)
    ]
    centered = [
        [gram[i][j] - (alpha if i == j else 0.0) for j in range(feature_dim)]
        for i in range(feature_dim)
    ]
    scale = 4.0 / token_count
    gradient = [
        [
            scale
            * sum(
                rows[token][source] * centered[source][target]
                for source in range(feature_dim)
            )
            for target in range(feature_dim)
        ]
        for token in range(token_count)
    ]
    return penalty, alpha, gradient


def build_packed_response_predictor_mask(
    *,
    sequence_lengths: Sequence[int],
    prompt_lengths: Sequence[int],
    response_lengths: Sequence[int],
    response_loss_masks: Sequence[Sequence[Any]],
) -> list[bool]:
    """Align response-token PPO masks with packed causal hidden states.

    veRL v0.9.0 keeps ``loss_mask`` in response-only coordinates even after
    ``input_ids`` becomes a packed nested tensor.  For response token ``j``
    of one sequence, causal-LM log probability comes from hidden position
    ``prompt_length - 1 + j``.  Construct that predictor-position mask per
    sequence so a shift can never cross a packed sequence boundary.
    """
    batch_size = len(sequence_lengths)
    if not (
        len(prompt_lengths)
        == len(response_lengths)
        == len(response_loss_masks)
        == batch_size
    ):
        raise ValueError("packed response metadata batch dimensions disagree")

    packed_mask: list[bool] = []
    for sample_index, (
        sequence_length,
        prompt_length,
        response_length,
        row,
    ) in enumerate(
        zip(
            sequence_lengths,
            prompt_lengths,
            response_lengths,
            response_loss_masks,
            strict=True,
        )
    ):
        sequence_length = int(sequence_length)
        prompt_length = int(prompt_length)
        response_length = int(response_length)
        row = list(row)

        if sequence_length <= 0:
            raise ValueError(f"sample {sample_index}: sequence length must be positive")
        if prompt_length <= 0:
            raise ValueError(
                f"sample {sample_index}: prompt length must be positive for causal alignment"
            )
        if response_length < 0:
            raise ValueError(
                f"sample {sample_index}: response length must be non-negative"
            )
        if prompt_length + response_length != sequence_length:
            raise ValueError(
                f"sample {sample_index}: prompt + response lengths do not equal "
                f"packed sequence length ({prompt_length} + {response_length} "
                f"!= {sequence_length})"
            )
        if len(row) < response_length:
            raise ValueError(
                f"sample {sample_index}: response loss mask is shorter than the "
                f"response ({len(row)} < {response_length})"
            )
        if any(bool(value) for value in row[response_length:]):
            raise ValueError(
                f"sample {sample_index}: masked token beyond response length"
            )

        sequence_mask = [False] * sequence_length
        predictor_start = prompt_length - 1
        for response_index, include in enumerate(row[:response_length]):
            if bool(include):
                sequence_mask[predictor_start + response_index] = True
        packed_mask.extend(sequence_mask)

    return packed_mask


def select_target_layer_names(
    module_names: Iterable[str], *, layer_mode: str, expected_hidden_layers: int
) -> list[str]:
    """Select Qwen dense MLP ``down_proj`` modules in transformer order."""
    indexed: dict[int, str] = {}
    for name in module_names:
        # FSDP inserts an exact ``_fsdp_wrapped_module`` path segment at the
        # root and inside every wrapped transformer block.  Strip only that
        # framework-owned segment, then require the observed Qwen language-
        # model path exactly; this cannot accidentally select the visual MLP.
        canonical_name = ".".join(
            segment for segment in name.split(".") if segment != _FSDP_WRAPPER_SEGMENT
        )
        match = _LAYER_PATTERN.fullmatch(canonical_name)
        if not match:
            continue
        layer_index = int(match.group(1))
        if layer_index in indexed:
            raise ValueError(
                f"multiple down_proj modules found for transformer layer {layer_index}"
            )
        indexed[layer_index] = name

    expected_indices = list(range(expected_hidden_layers))
    if sorted(indexed) != expected_indices:
        raise ValueError(
            "Qwen MLP layer inventory drifted: "
            f"expected={expected_indices}, actual={sorted(indexed)}"
        )
    ordered = [indexed[index] for index in expected_indices]
    if layer_mode == "last":
        return ordered[-1:]
    if layer_mode == "all_hidden":
        return ordered
    raise ValueError(f"unsupported layer_mode: {layer_mode}")
