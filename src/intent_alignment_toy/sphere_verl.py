"""veRL v0.9.0 integration for the dense Qwen SPHERE experiment arms.

This module intentionally leaves veRL's rollout and agent-loop paths untouched.
It replaces only the actor training worker and the synchronous trainer's actor
worker mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray
import torch
from codetiming import Timer
from tensordict import TensorDict
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from intent_alignment_toy.sphere_contract import (
    SphereVariant,
    build_packed_response_predictor_mask,
    load_sphere_variant,
    select_target_layer_names,
    sphere_penalty_from_gram_stats,
)

_EVIDENCE_SCHEMA = "intent_alignment_dense_sphere_worker_evidence_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a local tensor for regular tensors or single-rank DTensors."""
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        tensor = to_local()
    return tensor


def _tensor_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _local_tensor(tensor.detach())
    return torch.linalg.vector_norm(tensor, ord=2, dtype=torch.float32)


def _torch_sphere_penalty_and_feature_gradient(
    features: torch.Tensor, *, fixed_alpha: float | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact logical-minibatch SPHERE value and ``dL/dPhi``.

    The input is detached from the model graph.  Returning the closed-form
    feature gradient lets the worker replay physical microbatches with a
    linear surrogate, obtaining the exact logical-minibatch parameter gradient
    without retaining every transformer activation graph at once.
    """
    if features.ndim != 2 or min(features.shape) <= 0:
        raise ValueError(
            f"expected non-empty [T,D] features, got {tuple(features.shape)}"
        )
    phi = features.float()
    token_count, feature_dim = phi.shape
    if feature_dim <= token_count:
        gram = phi.transpose(0, 1) @ phi / float(token_count)
        trace = torch.trace(gram)
        frobenius_squared = torch.sum(gram * gram)
        penalty, alpha = sphere_penalty_from_gram_stats(
            frobenius_squared=frobenius_squared,
            trace=trace,
            feature_dim=feature_dim,
            fixed_alpha=fixed_alpha,
        )
        centered_gram = gram.clone()
        centered_gram.diagonal().sub_(alpha)
        feature_gradient = (4.0 / float(token_count)) * (phi @ centered_gram)
    else:
        gram = phi @ phi.transpose(0, 1) / float(token_count)
        trace = torch.trace(gram)
        frobenius_squared = torch.sum(gram * gram)
        penalty, alpha = sphere_penalty_from_gram_stats(
            frobenius_squared=frobenius_squared,
            trace=trace,
            feature_dim=feature_dim,
            fixed_alpha=fixed_alpha,
        )
        feature_gradient = (4.0 / float(token_count)) * (gram @ phi - alpha * phi)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.as_tensor(alpha, dtype=penalty.dtype, device=penalty.device)
    feature_gradient = feature_gradient.to(dtype=features.dtype)
    return penalty.detach(), alpha.detach(), feature_gradient.detach()


class SphereTrainingWorker(TrainingWorker):
    """Training worker that performs official-style gradient-ratio SPHERE."""

    def __init__(self, config):
        super().__init__(config)
        registry_path = Path(os.environ["IATOY_SPHERE_CONFIG_PATH"])
        variant_name = os.environ["IATOY_SPHERE_VARIANT"]
        self.sphere_variant: SphereVariant = load_sphere_variant(
            registry_path, variant_name
        )
        self.sphere_registry_path = registry_path
        self.sphere_evidence_path = Path(os.environ["IATOY_SPHERE_EVIDENCE_PATH"])
        self._hook_handles: list[Any] = []
        self._target_modules: dict[str, torch.nn.Module] = {}
        self._captures: dict[str, torch.Tensor] = {}
        self._capture_enabled = False
        self._capture_phase = "disabled"
        self._phase_microbatch_index = 0
        self._phase_stats: dict[str, dict[str, Any]] = {}
        self._logical_feature_chunks: dict[str, list[torch.Tensor]] = {}
        self._logical_gradient_chunks: dict[str, list[torch.Tensor]] = {}
        self._logical_microbatch_identities: list[str] = []
        self._logical_objective: dict[str, Any] | None = None
        self._logical_surrogate_correction: torch.Tensor | None = None
        self._evidence_activations: dict[str, torch.Tensor] = {}
        self._optimizer_steps = 0

    def reset(self):
        super().reset()
        self._install_feature_hooks()

    def _install_feature_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._target_modules.clear()
        self._captures.clear()

        module = self.engine.module
        named_modules = dict(module.named_modules())
        target_names = select_target_layer_names(
            named_modules,
            layer_mode=self.sphere_variant.layer_mode,
            expected_hidden_layers=self.sphere_variant.expected_hidden_layers,
        )
        for name in target_names:
            target = named_modules[name]
            if not hasattr(target, "weight"):
                raise RuntimeError(f"SPHERE target has no weight parameter: {name}")
            self._target_modules[name] = target

            def capture_hook(_module, inputs, *, layer_name=name):
                if not self._capture_enabled:
                    return
                if layer_name in self._captures:
                    raise RuntimeError(
                        f"SPHERE target executed more than once before loss: {layer_name}"
                    )
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise RuntimeError(
                        f"SPHERE target input is not a tensor: {layer_name}"
                    )
                feature = inputs[0]
                if (
                    self._capture_phase == "sphere"
                    and self._phase_microbatch_index == 0
                ):
                    feature.retain_grad()
                    self._evidence_activations[layer_name] = feature
                self._captures[layer_name] = feature

            self._hook_handles.append(target.register_forward_pre_hook(capture_hook))

        if len(self._target_modules) != (
            1
            if self.sphere_variant.layer_mode == "last"
            else self.sphere_variant.expected_hidden_layers
        ):
            raise RuntimeError(
                "SPHERE target layer count does not match the frozen variant"
            )

    def _begin_capture_phase(self, phase: str) -> None:
        if phase not in {"collect", "sphere", "combined"}:
            raise ValueError(f"unsupported SPHERE capture phase: {phase}")
        if self._capture_enabled or self._captures:
            raise RuntimeError("stale SPHERE captures exist at phase boundary")
        if phase == "collect":
            if self._logical_objective is not None or self._logical_gradient_chunks:
                raise RuntimeError("SPHERE logical-minibatch plan already exists")
            self._logical_feature_chunks = {name: [] for name in self._target_modules}
            self._logical_microbatch_identities = []
        elif self._logical_objective is None or not self._logical_gradient_chunks:
            raise RuntimeError("SPHERE logical-minibatch plan is not finalized")

        self._capture_phase = phase
        self._capture_enabled = True
        self._phase_microbatch_index = 0
        self._phase_stats[phase] = {
            "microbatches": 0,
            "identity_matches": 0,
            "surrogate_sum": None,
            "layers": {
                name: {
                    "calls": 0,
                    "token_rows": 0,
                    "feature_dim": None,
                    "surrogate_sum": None,
                }
                for name in self._target_modules
            },
        }
        if phase == "sphere":
            self._evidence_activations.clear()

    def _end_capture_phase(self) -> None:
        if self._captures:
            raise RuntimeError("SPHERE captures were not consumed at phase end")
        stats = self._phase_stats[self._capture_phase]
        microbatches = int(stats["microbatches"])
        if microbatches <= 0:
            raise RuntimeError(
                f"SPHERE phase had no microbatches: {self._capture_phase}"
            )
        if self._capture_phase == "collect":
            if len(self._logical_microbatch_identities) != microbatches:
                raise RuntimeError("SPHERE collection identity count drifted")
        else:
            expected = len(self._logical_microbatch_identities)
            if microbatches != expected or stats["identity_matches"] != expected:
                raise RuntimeError(
                    f"SPHERE replay coverage drifted in {self._capture_phase}: "
                    f"microbatches={microbatches}, identities={stats['identity_matches']}, "
                    f"expected={expected}"
                )
        self._capture_enabled = False
        self._capture_phase = "disabled"

    @staticmethod
    def _shifted_loss_mask(data: TensorDict, feature_rows: int) -> torch.Tensor:
        """Map response-only PPO loss rows to packed causal predictor rows."""
        input_ids = tu.get(data, key="input_ids")
        if not getattr(input_ids, "is_nested", False):
            raise RuntimeError("SPHERE expects veRL NO_PADDING nested input_ids")
        sequence_lengths = input_ids.offsets().diff()

        prompt_ids = tu.get(data, key="prompts")
        response_ids = tu.get(data, key="responses")
        if getattr(prompt_ids, "is_nested", False):
            if not getattr(response_ids, "is_nested", False):
                raise RuntimeError("SPHERE prompt/response layouts disagree")
            prompt_lengths = prompt_ids.offsets().diff()
            response_lengths = response_ids.offsets().diff()
        else:
            attention_mask = tu.get(data, key="attention_mask")
            if getattr(attention_mask, "is_nested", False):
                raise RuntimeError("SPHERE expects a dense source attention_mask")
            if attention_mask.ndim != 2 or prompt_ids.ndim != 2:
                raise RuntimeError("SPHERE prompt/attention tensors must be rank two")
            prompt_width = int(prompt_ids.shape[1])
            prompt_lengths = attention_mask[:, :prompt_width].sum(dim=1)
            response_lengths = attention_mask[:, prompt_width:].sum(dim=1)

        loss_mask = tu.get(data, key="loss_mask")
        if getattr(loss_mask, "is_nested", False):
            mask_lengths = loss_mask.offsets().diff()
            if not torch.equal(
                mask_lengths.detach().cpu(), response_lengths.detach().cpu()
            ):
                raise RuntimeError(
                    "SPHERE nested loss_mask is not response-only aligned"
                )
            response_loss_masks = [
                row.detach().cpu().tolist() for row in loss_mask.unbind()
            ]
            mask_device = loss_mask.values().device
        else:
            if loss_mask.ndim != 2 or loss_mask.shape[0] != sequence_lengths.shape[0]:
                raise RuntimeError(
                    "SPHERE loss_mask must have shape [batch, max_response_len]"
                )
            response_loss_masks = loss_mask.detach().cpu().tolist()
            mask_device = loss_mask.device

        packed_mask_values = build_packed_response_predictor_mask(
            sequence_lengths=sequence_lengths.detach().cpu().tolist(),
            prompt_lengths=prompt_lengths.detach().cpu().tolist(),
            response_lengths=response_lengths.detach().cpu().tolist(),
            response_loss_masks=response_loss_masks,
        )
        if len(packed_mask_values) != feature_rows:
            raise RuntimeError(
                "SPHERE feature/packed-input row mismatch: "
                f"features={feature_rows}, packed_input={len(packed_mask_values)}"
            )
        mask = torch.tensor(packed_mask_values, dtype=torch.bool, device=mask_device)
        if not torch.any(mask):
            raise RuntimeError(
                "SPHERE physical microbatch has no action-token features"
            )
        return mask

    @staticmethod
    def _microbatch_identity(data: TensorDict, packed_mask: torch.Tensor) -> str:
        """Hash token order and the selected predictor rows for replay checks."""
        input_ids = tu.get(data, key="input_ids")
        if not getattr(input_ids, "is_nested", False):
            raise RuntimeError("SPHERE identity expects nested packed input_ids")
        payload = {
            "offsets": input_ids.offsets().detach().cpu().tolist(),
            "input_ids": input_ids.values().detach().cpu().reshape(-1).tolist(),
            "predictor_mask": packed_mask.detach().cpu().tolist(),
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _selected_captures(
        self, data: TensorDict
    ) -> tuple[dict[str, torch.Tensor], str]:
        expected = set(self._target_modules)
        actual = set(self._captures)
        if actual != expected:
            raise RuntimeError(
                f"SPHERE hook coverage drifted: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

        selected_by_layer: dict[str, torch.Tensor] = {}
        shared_mask: torch.Tensor | None = None
        shared_rows: int | None = None
        for name in self._target_modules:
            feature = self._captures[name]
            feature_dim = int(feature.shape[-1])
            rows = feature.reshape(-1, feature_dim)
            if shared_mask is None:
                shared_rows = int(rows.shape[0])
                shared_mask = self._shifted_loss_mask(data, shared_rows)
            elif rows.shape[0] != shared_rows:
                raise RuntimeError("SPHERE target layers expose different token counts")
            selected_by_layer[name] = rows[shared_mask]

        assert shared_mask is not None
        return selected_by_layer, self._microbatch_identity(data, shared_mask)

    @staticmethod
    def _accumulate_scalar(current, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        return value if current is None else current + value

    def _collect_logical_features(self, data: TensorDict) -> None:
        if self._capture_phase != "collect":
            raise RuntimeError("SPHERE feature collection ran outside collect phase")
        selected_by_layer, identity = self._selected_captures(data)
        stats = self._phase_stats["collect"]
        self._logical_microbatch_identities.append(identity)

        for name, selected in selected_by_layer.items():
            feature_dim = int(selected.shape[-1])
            self._logical_feature_chunks[name].append(selected.detach().clone())
            layer_stats = stats["layers"][name]
            layer_stats["calls"] += 1
            layer_stats["token_rows"] += int(selected.shape[0])
            if layer_stats["feature_dim"] not in (None, feature_dim):
                raise RuntimeError(f"SPHERE feature dimension drifted at {name}")
            layer_stats["feature_dim"] = feature_dim

        stats["microbatches"] += 1
        self._phase_microbatch_index += 1
        self._captures.clear()

    def _finalize_logical_objective(self) -> None:
        if self._capture_enabled or self._capture_phase != "disabled":
            raise RuntimeError("cannot finalize SPHERE plan during capture")
        if not self._logical_microbatch_identities:
            raise RuntimeError("SPHERE collected no physical microbatches")

        penalties: list[torch.Tensor] = []
        surrogate_dots: list[torch.Tensor] = []
        logical_layers: dict[str, dict[str, Any]] = {}
        expected_rows: list[int] | None = None
        self._logical_gradient_chunks = {}

        for name in self._target_modules:
            chunks = self._logical_feature_chunks.get(name, [])
            if len(chunks) != len(self._logical_microbatch_identities):
                raise RuntimeError(f"SPHERE feature chunk coverage drifted at {name}")
            chunk_rows = [int(chunk.shape[0]) for chunk in chunks]
            if expected_rows is None:
                expected_rows = chunk_rows
            elif chunk_rows != expected_rows:
                raise RuntimeError("SPHERE target layers expose different chunk rows")

            logical_features = torch.cat(chunks, dim=0)
            penalty, alpha, feature_gradient = (
                _torch_sphere_penalty_and_feature_gradient(
                    logical_features,
                    fixed_alpha=self.sphere_variant.fixed_alpha,
                )
            )
            if (
                not bool(torch.isfinite(penalty).item())
                or not bool(torch.isfinite(alpha).item())
                or not bool(torch.all(torch.isfinite(feature_gradient)).item())
            ):
                raise FloatingPointError(f"non-finite logical SPHERE value at {name}")
            feature_gradient_norm = _tensor_l2_norm(feature_gradient)
            if float(feature_gradient_norm.item()) <= 0:
                raise RuntimeError(f"logical SPHERE feature gradient is zero at {name}")

            gradient_chunks = [
                chunk.detach().clone()
                for chunk in torch.split(feature_gradient, chunk_rows, dim=0)
            ]
            if len(gradient_chunks) != len(chunks):
                raise RuntimeError(f"SPHERE gradient split drifted at {name}")
            surrogate_dot = torch.stack(
                [
                    torch.sum(feature.float() * gradient.float())
                    for feature, gradient in zip(chunks, gradient_chunks, strict=True)
                ]
            ).sum()

            self._logical_gradient_chunks[name] = gradient_chunks
            penalties.append(penalty)
            surrogate_dots.append(surrogate_dot.detach())
            logical_layers[name] = {
                "calls": len(chunks),
                "token_rows": int(logical_features.shape[0]),
                "feature_dim": int(logical_features.shape[1]),
                "penalty": penalty,
                "alpha": alpha,
                "feature_gradient_l2_norm": feature_gradient_norm.detach(),
                "surrogate_dot": surrogate_dot.detach(),
            }
            self._logical_feature_chunks[name] = []
            del logical_features, feature_gradient

        total_penalty = torch.stack(penalties).sum()
        total_surrogate_dot = torch.stack(surrogate_dots).sum()
        correction = (total_penalty - total_surrogate_dot).detach()
        if (
            not bool(torch.isfinite(total_penalty).item())
            or float(total_penalty.item()) <= 0
        ):
            raise RuntimeError("logical-minibatch SPHERE penalty is zero/non-finite")
        if not bool(torch.isfinite(correction).item()):
            raise RuntimeError(
                "logical-minibatch SPHERE surrogate correction is non-finite"
            )

        self._logical_feature_chunks = {}
        self._logical_surrogate_correction = correction
        self._logical_objective = {
            "microbatches": len(self._logical_microbatch_identities),
            "token_rows_per_microbatch": expected_rows,
            "penalty_sum": total_penalty.detach(),
            "surrogate_dot_sum": total_surrogate_dot.detach(),
            "surrogate_correction": correction,
            "layers": logical_layers,
        }

    def _consume_logical_surrogate(self, data: TensorDict) -> torch.Tensor:
        phase = self._capture_phase
        if phase not in {"sphere", "combined"}:
            raise RuntimeError("SPHERE surrogate ran outside a replay phase")
        if (
            self._logical_objective is None
            or self._logical_surrogate_correction is None
        ):
            raise RuntimeError("SPHERE logical-minibatch plan is missing")

        selected_by_layer, identity = self._selected_captures(data)
        index = self._phase_microbatch_index
        if index >= len(self._logical_microbatch_identities):
            raise RuntimeError(f"SPHERE {phase} replay has an extra microbatch")
        expected_identity = self._logical_microbatch_identities[index]
        if identity != expected_identity:
            raise RuntimeError(
                f"SPHERE {phase} microbatch identity drifted at index {index}: "
                f"expected={expected_identity}, actual={identity}"
            )

        stats = self._phase_stats[phase]
        terms: list[torch.Tensor] = []
        for name, selected in selected_by_layer.items():
            gradient = self._logical_gradient_chunks[name][index]
            if selected.shape != gradient.shape:
                raise RuntimeError(
                    f"SPHERE {phase} feature/gradient shape drifted at {name}: "
                    f"{tuple(selected.shape)} != {tuple(gradient.shape)}"
                )
            term = torch.sum(selected.float() * gradient.float())
            terms.append(term)
            layer_stats = stats["layers"][name]
            layer_stats["calls"] += 1
            layer_stats["token_rows"] += int(selected.shape[0])
            layer_stats["feature_dim"] = int(selected.shape[1])
            layer_stats["surrogate_sum"] = self._accumulate_scalar(
                layer_stats["surrogate_sum"], term
            )

        surrogate = torch.stack(terms).sum()
        if index == 0:
            surrogate = surrogate + self._logical_surrogate_correction
        if not bool(torch.isfinite(surrogate).item()):
            raise FloatingPointError(f"non-finite SPHERE surrogate in {phase}")

        stats["microbatches"] += 1
        stats["identity_matches"] += 1
        stats["surrogate_sum"] = self._accumulate_scalar(
            stats["surrogate_sum"], surrogate
        )
        self._phase_microbatch_index += 1
        self._captures.clear()
        return surrogate

    def _ppo_collect_loss(self, model_output, data: TensorDict, dp_group=None):
        base_loss, metrics = self.loss_fn(
            model_output=model_output, data=data, dp_group=dp_group
        )
        self._collect_logical_features(data)
        return base_loss, metrics

    def _sphere_only_loss(self, model_output, data: TensorDict, dp_group=None):
        # FSDP reshard-after-forward requires a backward edge from the wrapped
        # module output so its pre-backward hook can restore flattened parameter
        # storage.  SPHERE branches from an internal MLP activation, therefore
        # keep an exactly-zero PPO anchor in this pass instead of bypassing the
        # model output graph (which otherwise fails with a zero-sized FSDP
        # parameter storage during backward).
        base_loss, _ = self.loss_fn(
            model_output=model_output, data=data, dp_group=dp_group
        )
        surrogate = self._consume_logical_surrogate(data)
        anchored_surrogate = surrogate + base_loss * 0.0
        metrics = {
            "actor/sphere_penalty": Metric(
                value=surrogate, aggregation=AggregationType.SUM
            )
        }
        return anchored_surrogate, metrics

    def _combined_loss(self, model_output, data: TensorDict, dp_group=None):
        base_loss, metrics = self.loss_fn(
            model_output=model_output, data=data, dp_group=dp_group
        )
        surrogate = self._consume_logical_surrogate(data)
        total_loss = base_loss + self._current_lambda * surrogate
        metrics["actor/sphere_penalty"] = Metric(
            value=surrogate, aggregation=AggregationType.SUM
        )
        metrics["actor/sphere_lambda"] = Metric(
            value=self._current_lambda, aggregation=AggregationType.MEAN
        )
        return total_loss, metrics

    def _gradient_stats(self) -> dict[str, float | int]:
        norm_squared = torch.zeros((), dtype=torch.float32, device=self.device_name)
        projection = torch.zeros((), dtype=torch.float32, device=self.device_name)
        sampled_values = 0
        tensors_with_grad = 0
        finite = True

        for parameter_index, (_name, parameter) in enumerate(
            self.engine.module.named_parameters()
        ):
            if parameter.grad is None:
                continue
            grad = _local_tensor(parameter.grad.detach())
            tensors_with_grad += 1
            norm = torch.linalg.vector_norm(grad, ord=2, dtype=torch.float32)
            norm_squared = norm_squared + norm * norm
            finite = finite and bool(torch.isfinite(norm).item())

            flat = grad.reshape(-1)
            count = min(17, flat.numel())
            if count:
                # This projection is evidence only; it is not part of the
                # optimizer.  Use a basic first-N slice rather than CUDA
                # integer linspace + advanced indexing.  Torch 2.11/CUDA 13
                # rounds the final linspace index up to ``numel`` for several
                # real Qwen tensor sizes (for example 635699200), which makes
                # an otherwise valid backward fail with a device-side assert.
                sample = flat.narrow(0, 0, count).float()
                weights = (
                    (torch.arange(count, device=flat.device, dtype=torch.float32) + 1)
                    * float(parameter_index + 1)
                    / float(count)
                )
                projection = projection + torch.sum(sample * weights)
                sampled_values += count

        norm_value = float(torch.sqrt(norm_squared).item())
        projection_value = float(projection.item())
        if (
            not finite
            or not math.isfinite(norm_value)
            or not math.isfinite(projection_value)
        ):
            raise FloatingPointError("non-finite gradient statistics")
        return {
            "l2_norm": norm_value,
            "projection": projection_value,
            "tensors_with_grad": tensors_with_grad,
            "projection_sample_count": sampled_values,
        }

    def _activation_gradient_evidence(self) -> dict[str, dict[str, Any]]:
        evidence = {}
        for name in self._target_modules:
            activation = self._evidence_activations.get(name)
            if activation is None or activation.grad is None:
                raise RuntimeError(f"SPHERE activation gradient missing at {name}")
            norm = float(_tensor_l2_norm(activation.grad).item())
            finite = math.isfinite(norm)
            evidence[name] = {
                "shape": list(activation.shape),
                "gradient_l2_norm": norm,
                "gradient_finite": finite,
                "gradient_nonzero": finite and norm > 0,
            }
            if not evidence[name]["gradient_nonzero"]:
                raise RuntimeError(
                    f"SPHERE activation gradient is zero/non-finite at {name}"
                )
        self._evidence_activations.clear()
        return evidence

    @staticmethod
    def _serialize_collection_stats(stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "microbatches": int(stats["microbatches"]),
            "layers": {
                name: {
                    "calls": int(layer["calls"]),
                    "token_rows": int(layer["token_rows"]),
                    "feature_dim": int(layer["feature_dim"]),
                }
                for name, layer in stats["layers"].items()
            },
        }

    @staticmethod
    def _serialize_logical_objective(stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "microbatches": int(stats["microbatches"]),
            "token_rows_per_microbatch": list(stats["token_rows_per_microbatch"]),
            "penalty_sum": float(stats["penalty_sum"].item()),
            "surrogate_dot_sum": float(stats["surrogate_dot_sum"].item()),
            "surrogate_correction": float(stats["surrogate_correction"].item()),
            "layers": {
                name: {
                    "calls": int(layer["calls"]),
                    "token_rows": int(layer["token_rows"]),
                    "feature_dim": int(layer["feature_dim"]),
                    "penalty": float(layer["penalty"].item()),
                    "alpha": float(layer["alpha"].item()),
                    "feature_gradient_l2_norm": float(
                        layer["feature_gradient_l2_norm"].item()
                    ),
                    "surrogate_dot": float(layer["surrogate_dot"].item()),
                }
                for name, layer in stats["layers"].items()
            },
        }

    def _serialize_replay_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        if self._logical_objective is None:
            raise RuntimeError("SPHERE logical objective missing during serialization")
        penalty = float(self._logical_objective["penalty_sum"].item())
        surrogate = float(stats["surrogate_sum"].item())
        return {
            "microbatches": int(stats["microbatches"]),
            "identity_matches": int(stats["identity_matches"]),
            "penalty_sum": penalty,
            "surrogate_sum": surrogate,
            "surrogate_matches_penalty": math.isclose(
                surrogate, penalty, rel_tol=5e-4, abs_tol=1e-2
            ),
            "layers": {
                name: {
                    "calls": int(layer["calls"]),
                    "token_rows": int(layer["token_rows"]),
                    "feature_dim": int(layer["feature_dim"]),
                    "surrogate_sum_before_global_correction": float(
                        layer["surrogate_sum"].item()
                    ),
                }
                for name, layer in stats["layers"].items()
            },
        }

    def _write_evidence(
        self,
        *,
        base_gradient: dict[str, Any],
        sphere_gradient: dict[str, Any],
        final_gradient: dict[str, Any],
        activation_gradients: dict[str, Any],
        optimizer_grad_norm: float,
    ) -> None:
        if self._logical_objective is None:
            raise RuntimeError("SPHERE logical objective missing at evidence write")
        sphere_norm = float(sphere_gradient["l2_norm"])
        base_norm = float(base_gradient["l2_norm"])
        contribution_norm = self._current_lambda * sphere_norm
        ratio = contribution_norm / base_norm
        expected_lambda = (
            self.sphere_variant.rho
            * base_norm
            / (sphere_norm + self.sphere_variant.epsilon)
        )
        expected_ratio = expected_lambda * sphere_norm / base_norm
        predicted_projection = float(base_gradient["projection"]) + (
            self._current_lambda * float(sphere_gradient["projection"])
        )
        actual_projection = float(final_gradient["projection"])
        projection_residual = actual_projection - predicted_projection
        projection_tolerance = max(
            1e-5,
            5e-4 * max(abs(predicted_projection), abs(actual_projection), 1.0),
        )
        registry_sha256 = _sha256(self.sphere_registry_path)
        logical = self._serialize_logical_objective(self._logical_objective)
        sphere_replay = self._serialize_replay_stats(self._phase_stats["sphere"])
        combined_replay = self._serialize_replay_stats(self._phase_stats["combined"])
        optimizer_grad_norm = float(optimizer_grad_norm)

        payload = {
            "schema": _EVIDENCE_SCHEMA,
            "created_at_utc": _utc_now(),
            "variant": self.sphere_variant.name,
            "contract": {
                "layer_mode": self.sphere_variant.layer_mode,
                "target_scale_mode": self.sphere_variant.target_scale_mode,
                "fixed_alpha": self.sphere_variant.fixed_alpha,
                "coefficient_mode": self.sphere_variant.coefficient_mode,
                "rho": self.sphere_variant.rho,
                "epsilon": self.sphere_variant.epsilon,
                "feature_site": self.sphere_variant.feature_site,
                "token_selection": self.sphere_variant.token_selection,
                "layer_aggregation": self.sphere_variant.layer_aggregation,
                "physical_microbatch_penalty": False,
                "logical_minibatch_exact_penalty": True,
                "logical_minibatch_surrogate_replay": True,
                "logical_minibatch_gradient_ratio": True,
                "gradient_checkpointing_required_off": True,
                "torch_compile_required_off": True,
            },
            "registry": {
                "path": str(self.sphere_registry_path),
                "sha256": registry_sha256,
            },
            "runtime": {
                "torch": torch.__version__,
                "world_size": torch.distributed.get_world_size(),
                "rank": torch.distributed.get_rank(),
                "engine": type(self.engine).__name__,
                "model_wrapper": type(self.engine.module).__name__,
            },
            "targets": {
                "count": len(self._target_modules),
                "names": list(self._target_modules),
            },
            "passes": {
                "ppo_only_forward_backward": 1,
                "sphere_only_forward_backward": 1,
                "combined_forward_backward": 1,
                "optimizer_steps": self._optimizer_steps,
                "collection": self._serialize_collection_stats(
                    self._phase_stats["collect"]
                ),
                "logical_objective": logical,
                "sphere": sphere_replay,
                "combined": combined_replay,
            },
            "gradients": {
                "ppo": base_gradient,
                "sphere": sphere_gradient,
                "combined": final_gradient,
                "lambda": self._current_lambda,
                "expected_lambda": expected_lambda,
                "regularizer_contribution_l2_norm": contribution_norm,
                "regularizer_to_ppo_norm_ratio": ratio,
                "expected_regularizer_to_ppo_norm_ratio": expected_ratio,
                "predicted_combined_projection": predicted_projection,
                "actual_combined_projection": actual_projection,
                "combined_projection_residual": projection_residual,
                "combined_projection_tolerance": projection_tolerance,
                "optimizer_reported_preclip_grad_norm": optimizer_grad_norm,
                "target_activation_gradients": activation_gradients,
            },
            "checks": {
                "logical_minibatch_penalty_finite_nonzero": (
                    math.isfinite(logical["penalty_sum"]) and logical["penalty_sum"] > 0
                ),
                "logical_feature_gradients_finite_nonzero": all(
                    math.isfinite(layer["feature_gradient_l2_norm"])
                    and layer["feature_gradient_l2_norm"] > 0
                    for layer in logical["layers"].values()
                ),
                "sphere_gradient_finite_nonzero": (
                    math.isfinite(sphere_norm) and sphere_norm > 0
                ),
                "lambda_formula_matches": math.isclose(
                    self._current_lambda,
                    expected_lambda,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ),
                "ratio_formula_matches": math.isclose(
                    ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-15
                ),
                "sphere_replay_matches_logical_penalty": sphere_replay[
                    "surrogate_matches_penalty"
                ],
                "combined_replay_matches_logical_penalty": combined_replay[
                    "surrogate_matches_penalty"
                ],
                "all_replay_microbatch_identities_match": (
                    sphere_replay["identity_matches"]
                    == combined_replay["identity_matches"]
                    == logical["microbatches"]
                ),
                "combined_projection_matches_sum": (
                    math.isfinite(projection_residual)
                    and abs(projection_residual) <= projection_tolerance
                ),
                "optimizer_preclip_grad_norm_finite_nonzero": (
                    math.isfinite(optimizer_grad_norm) and optimizer_grad_norm > 0
                ),
                "all_target_activation_gradients_nonzero": all(
                    item["gradient_nonzero"] for item in activation_gradients.values()
                ),
                "optimizer_step_exactly_one": self._optimizer_steps == 1,
            },
        }
        if not all(payload["checks"].values()):
            raise RuntimeError(f"SPHERE evidence gate failed: {payload['checks']}")

        self.sphere_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.sphere_evidence_path.with_name(
            f".{self.sphere_evidence_path.name}.tmp.{os.getpid()}"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.sphere_evidence_path)
        self._logical_gradient_chunks = {}
        self._logical_surrogate_correction = None

    def _add_global_metric(self, output: dict, name: str, value: float) -> None:
        output["metrics"][name] = [
            Metric(value=value, aggregation=AggregationType.MEAN)
        ]

    def train_batch(self, data: TensorDict) -> TensorDict:
        if self._optimizer_steps != 0:
            raise RuntimeError(
                "frozen SPHERE toy contract permits exactly one optimizer step"
            )
        if torch.distributed.get_world_size() != 1:
            raise RuntimeError("dense SPHERE toy contract is frozen to one GPU")
        if self.loss_fn is None:
            raise RuntimeError(
                "loss function must be configured before SPHERE training"
            )
        if self.engine_config.forward_only:
            raise RuntimeError("SPHERE training cannot use a forward-only engine")

        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)
        default_keys = {
            "use_remove_padding": self.model_config.get("use_remove_padding", False),
            "use_dynamic_bsz": self.engine_config.use_dynamic_bsz,
            "max_token_len_per_gpu": self.engine_config.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.engine_config.micro_batch_size_per_gpu,
            "use_fused_kernels": self.engine_config.use_fused_kernels,
        }
        for key, value in default_keys.items():
            if key not in data:
                tu.assign_non_tensor(data, **{key: value})
        maybe_fix_3d_position_ids(data)

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            self.engine.optimizer_zero_grad()
            self._begin_capture_phase("collect")
            self.engine.forward_backward_batch(
                data, loss_function=self._ppo_collect_loss, forward_only=False
            )
            self._end_capture_phase()
            base_gradient = self._gradient_stats()
            if base_gradient["l2_norm"] <= 0:
                raise RuntimeError("PPO gradient norm is zero")
            self._finalize_logical_objective()

            self.engine.optimizer_zero_grad()
            self._begin_capture_phase("sphere")
            self.engine.forward_backward_batch(
                data, loss_function=self._sphere_only_loss, forward_only=False
            )
            self._end_capture_phase()
            sphere_gradient = self._gradient_stats()
            activation_gradients = self._activation_gradient_evidence()
            if sphere_gradient["l2_norm"] <= 0:
                raise RuntimeError("SPHERE gradient norm is zero")

            self._current_lambda = (
                self.sphere_variant.rho
                * float(base_gradient["l2_norm"])
                / (float(sphere_gradient["l2_norm"]) + self.sphere_variant.epsilon)
            )
            if not math.isfinite(self._current_lambda) or self._current_lambda <= 0:
                raise FloatingPointError(
                    "SPHERE adaptive coefficient is non-finite/non-positive"
                )

            self.engine.optimizer_zero_grad()
            self._begin_capture_phase("combined")
            output = self.engine.forward_backward_batch(
                data, loss_function=self._combined_loss, forward_only=False
            )
            self._end_capture_phase()
            final_gradient = self._gradient_stats()
            optimizer_grad_norm = self.engine.optimizer_step()
            self._optimizer_steps += 1

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        lr = self.engine.lr_scheduler_step() if update_lr_scheduler else None
        if self.engine.is_mp_src_rank_with_outputs():
            output.pop("model_output")
            output["metrics"]["grad_norm"] = optimizer_grad_norm
            if lr is not None:
                output["metrics"]["lr"] = lr
            self._add_global_metric(
                output, "actor/sphere_ppo_grad_norm", base_gradient["l2_norm"]
            )
            self._add_global_metric(
                output, "actor/sphere_grad_norm", sphere_gradient["l2_norm"]
            )
            self._add_global_metric(output, "actor/sphere_lambda", self._current_lambda)
            self._add_global_metric(
                output,
                "actor/sphere_contribution_ratio",
                self._current_lambda
                * sphere_gradient["l2_norm"]
                / base_gradient["l2_norm"],
            )
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=timer.last,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
            self._write_evidence(
                base_gradient=base_gradient,
                sphere_gradient=sphere_gradient,
                final_gradient=final_gradient,
                activation_gradients=activation_gradients,
                optimizer_grad_norm=optimizer_grad_norm,
            )
        else:
            final_output = None
        return final_output


class SphereActorRolloutRefWorker(ActorRolloutRefWorker):
    """veRL hybrid worker with SPHERE only on the trainable actor."""

    actor_worker_cls = SphereTrainingWorker


class SpherePPOTrainerSync(PPOTrainerSync):
    """Synchronous V1 PPO trainer that swaps only the actor worker class."""

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        replaced = 0
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(SphereActorRolloutRefWorker)
                replaced += 1
        if replaced != 1:
            raise RuntimeError(
                f"expected exactly one actor/rollout role replacement, got {replaced}"
            )
