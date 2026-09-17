import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_sphere_training import (
    EXPECTED_TEST_JSONL_SHA256,
    audit_final_run,
    audit_training_run,
    sha256_file,
    validate_worker_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "sphere_variants.json"
TARGET = (
    "_fsdp_wrapped_module.model.language_model.layers.31."
    "_fsdp_wrapped_module.mlp.down_proj"
)


def worker_payload():
    ppo_norm = 2.0
    sphere_norm = 4.0
    rho = 1e-3
    epsilon = 1e-8
    coefficient = rho * ppo_norm / (sphere_norm + epsilon)
    ratio = coefficient * sphere_norm / ppo_norm
    ppo_projection = 0.5
    sphere_projection = 0.25
    assembled_projection = ppo_projection + coefficient * sphere_projection
    ppo_sphere_dot = 0.0
    assembled_norm = (
        ppo_norm * ppo_norm
        + 2.0 * coefficient * ppo_sphere_dot
        + coefficient * coefficient * sphere_norm * sphere_norm
    ) ** 0.5

    def gradient(norm, projection):
        return {
            "l2_norm": norm,
            "projection": projection,
            "tensors_with_grad": 3,
            "projection_sample_count": 17,
        }

    replay_layer = {
        "calls": 2,
        "token_rows": 4,
        "feature_dim": 9216,
        "surrogate_sum_before_global_correction": 12.0,
        "feature_difference_l2_norm": 0.0,
        "feature_reference_l2_norm": 100.0,
        "feature_relative_l2_difference": 0.0,
        "feature_max_abs_difference": 0.0,
        "feature_reference_max_abs": 2.0,
        "feature_relative_max_difference": 0.0,
        "feature_different_values": 0,
        "feature_total_values": 4 * 9216,
        "feature_replay_within_tolerance": True,
    }
    replay = {
        "microbatches": 2,
        "identity_matches": 2,
        "penalty_sum": 8.0,
        "collection_surrogate_dot_sum": 12.0,
        "raw_surrogate_sum": 12.0,
        "raw_surrogate_difference": 0.0,
        "raw_surrogate_relative_difference": 0.0,
        "surrogate_sum": 8.0,
        "surrogate_matches_penalty": True,
        "feature_replay": {
            "difference_l2_norm": 0.0,
            "reference_l2_norm": 100.0,
            "relative_l2_difference": 0.0,
            "max_abs_difference": 0.0,
            "reference_max_abs": 2.0,
            "relative_max_difference": 0.0,
            "different_values": 0,
            "total_values": 4 * 9216,
            "relative_l2_tolerance": 1e-5,
            "relative_max_tolerance": 1e-3,
            "within_tolerance": True,
        },
        "layers": {TARGET: replay_layer},
    }
    checks = {
        "logical_minibatch_penalty_finite_nonzero": True,
        "logical_feature_gradients_finite_nonzero": True,
        "sphere_gradient_finite_nonzero": True,
        "lambda_formula_matches": True,
        "ratio_formula_matches": True,
        "sphere_replay_matches_logical_penalty": True,
        "sphere_replay_features_within_tolerance": True,
        "all_replay_microbatch_identities_match": True,
        "optimizer_gradient_assembly_finite": True,
        "full_tensor_assembly_exact": True,
        "postcast_regularizer_contribution_nonzero": True,
        "optimizer_projection_matches_explicit_sum": True,
        "optimizer_preclip_grad_norm_finite_nonzero": True,
        "all_target_activation_gradients_nonzero": True,
        "optimizer_step_exactly_one": True,
    }
    return {
        "schema": "intent_alignment_dense_sphere_worker_evidence_v4",
        "status": "passed",
        "variant": "sphere_dense_last_adaptive_v1",
        "run": {
            "id": "test-run",
            "nonce": "a" * 64,
            "evidence_path": "/tmp/test-run/sphere/worker-evidence.json",
        },
        "contract": {
            "layer_mode": "last",
            "target_scale_mode": "adaptive_trace",
            "fixed_alpha": None,
            "coefficient_mode": "gradient_norm_ratio",
            "rho": rho,
            "epsilon": epsilon,
            "feature_site": "mlp.down_proj.input",
            "token_selection": "shifted_ppo_loss_mask",
            "layer_aggregation": "sum",
            "physical_microbatch_penalty": False,
            "logical_minibatch_exact_penalty": True,
            "logical_minibatch_surrogate_replay": True,
            "logical_minibatch_gradient_ratio": True,
            "optimizer_gradient_assembly": "explicit_ppo_plus_lambda_sphere",
            "gradient_checkpointing_required_off": True,
            "torch_compile_required_off": True,
        },
        "registry": {"sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest()},
        "runtime": {"world_size": 1, "rank": 0},
        "targets": {"count": 1, "names": [TARGET]},
        "passes": {
            "ppo_only_forward_backward": 1,
            "sphere_only_forward_backward": 1,
            "combined_forward_backward": 0,
            "explicit_gradient_combination": 1,
            "optimizer_steps": 1,
            "collection": {
                "microbatches": 2,
                "layers": {
                    TARGET: {
                        "calls": 2,
                        "token_rows": 4,
                        "feature_dim": 9216,
                    }
                },
            },
            "logical_objective": {
                "microbatches": 2,
                "token_rows_per_microbatch": [2, 2],
                "penalty_sum": 8.0,
                "surrogate_dot_sum": 12.0,
                "surrogate_correction": -4.0,
                "layers": {
                    TARGET: {
                        "calls": 2,
                        "token_rows": 4,
                        "feature_dim": 9216,
                        "penalty": 8.0,
                        "alpha": 0.25,
                        "feature_gradient_l2_norm": 3.0,
                        "surrogate_dot": 12.0,
                    }
                },
            },
            "sphere": replay,
        },
        "gradient_assembly": {
            "mode": "explicit_ppo_plus_lambda_sphere",
            "parameter_tensors": 3,
            "element_count": 100,
            "finite": True,
            "accumulation_dtype": "torch.float32",
            "ppo_source_dtype_counts": {"torch.float32": 3},
            "sphere_source_dtype_counts": {"torch.float32": 3},
            "destination_dtype_counts": {"torch.float32": 3},
            "ppo_representation_counts": {"torch.Tensor": 3},
            "sphere_representation_counts": {"torch.Tensor": 3},
            "destination_representation_counts": {"torch.Tensor": 3},
            "ppo_placement_counts": {"regular_tensor": 3},
            "sphere_placement_counts": {"regular_tensor": 3},
            "destination_placement_counts": {"regular_tensor": 3},
            "ppo_l2_norm": ppo_norm,
            "sphere_l2_norm": sphere_norm,
            "ppo_sphere_dot_product": ppo_sphere_dot,
            "precast_expected_l2_norm": assembled_norm,
            "postcast_expected_l2_norm": assembled_norm,
            "actual_l2_norm": assembled_norm,
            "full_tensor_residual_l2_norm": 0.0,
            "full_tensor_residual_max_abs": 0.0,
            "postcast_delta_from_ppo_l2_norm": coefficient * sphere_norm,
            "postcast_delta_from_ppo_max_abs": coefficient,
            "postcast_values_different_from_ppo": 2,
            "postcast_quantization_error_l2_norm": 0.0,
            "postcast_quantization_error_max_abs": 0.0,
        },
        "gradients": {
            "ppo": gradient(ppo_norm, ppo_projection),
            "sphere": gradient(sphere_norm, sphere_projection),
            "optimizer_assembled": gradient(assembled_norm, assembled_projection),
            "lambda": coefficient,
            "expected_lambda": coefficient,
            "regularizer_contribution_l2_norm": coefficient * sphere_norm,
            "regularizer_to_ppo_norm_ratio": ratio,
            "expected_regularizer_to_ppo_norm_ratio": ratio,
            "predicted_optimizer_projection": assembled_projection,
            "actual_optimizer_projection": assembled_projection,
            "optimizer_projection_residual": 0.0,
            "optimizer_projection_tolerance": 5e-4,
            "optimizer_reported_preclip_grad_norm": assembled_norm,
            "target_activation_gradients": {
                TARGET: {
                    "gradient_finite": True,
                    "gradient_nonzero": True,
                    "gradient_l2_norm": 1.0,
                }
            },
        },
        "checks": checks,
    }


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _resolved_config():
    return """\
model_engine: dp
data:
  train_batch_size: 400
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: false
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 400
    ppo_micro_batch_size_per_gpu: 4
    ppo_epochs: 1
    use_torch_compile: false
    fsdp_config:
      strategy: fsdp
      model_dtype: fp32
      dtype: bfloat16
      use_orig_params: false
      use_torch_compile: false
  rollout:
    n: 2
trainer:
  total_training_steps: 1
  save_freq: 1
  resume_mode: disable
"""


def _make_training_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    nonce_path = run / "sphere/run-nonce"
    nonce_path.parent.mkdir(parents=True)
    nonce_path.write_text("a" * 64 + "\n", encoding="utf-8")
    worker = worker_payload()
    worker["run"] = {
        "id": run.name,
        "nonce": "a" * 64,
        "evidence_path": str(run / "sphere/worker-evidence.json"),
    }
    _write_json(run / "sphere/worker-evidence.json", worker)
    rows = [
        {"input": "a", "score": 1.0, "step": 1},
        {"input": "a", "score": 0.0, "step": 1},
        {"input": "b", "score": 1.0, "step": 1},
        {"input": "b", "score": 1.0, "step": 1},
    ]
    (run / "rollouts").mkdir(parents=True)
    (run / "rollouts/1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    checkpoint = run / "checkpoints/global_step_1"
    hf = checkpoint / "actor/huggingface"
    hf.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    (hf / "config.json").write_text("{}\n", encoding="utf-8")
    (hf / "model.safetensors").write_bytes(b"weights")
    parameter = {
        "schema": "intent_alignment_semantic_parameter_evidence_v2",
        "after_path": str(hf),
        "after_config_sha256": sha256_file(hf / "config.json"),
        "changed_probe_count": 1,
        "weights_changed": True,
        "effective_bfloat16_weights_changed": True,
        "evaluated_weight_files": {
            "after": [
                {
                    "name": "model.safetensors",
                    "sha256": sha256_file(hf / "model.safetensors"),
                }
            ]
        },
    }
    _write_json(run / "parameter-evidence.json", parameter)
    contract = {
        "schema": "intent_alignment_dense_sphere_run_contract_v3",
        "run_id": run.name,
        "run_nonce_sha256": sha256_file(nonce_path),
        "variant": "sphere_dense_last_adaptive_v1",
        "constraints": {
            "no_intermediate_evaluation": True,
            "restore_holder_after_run": True,
            "do_not_release_allocation": True,
            "no_new_allocation": True,
        },
        "algorithm": {
            "base": "GRPO",
            "sphere_variant": "sphere_dense_last_adaptive_v1",
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
        },
        "runtime": {"verl_commit": "483b8a009ba3a97563edee3a19887e4862b8094a"},
        "data": {"test_jsonl_sha256": EXPECTED_TEST_JSONL_SHA256},
    }
    _write_json(run / "contract/run-contract.json", contract)
    (run / "contract/resolved-config.yaml").write_text(
        _resolved_config(), encoding="utf-8"
    )
    return run


def _evaluation(successes: int):
    tasks = [
        {"id": f"test-{index:03d}", "success": index < successes}
        for index in range(100)
    ]
    return {
        "protocol": {"dataset_sha256": EXPECTED_TEST_JSONL_SHA256},
        "summary": {
            "total": 100,
            "successes": successes,
            "success_rate": successes / 100,
        },
        "tasks": tasks,
    }


def test_validate_worker_evidence_recomputes_lambda_ratio_and_projection():
    result = validate_worker_evidence(
        worker_payload(),
        expected_variant="sphere_dense_last_adaptive_v1",
        registry_path=REGISTRY,
    )
    assert result["target_layer_indices"] == [31]
    assert result["logical_minibatch_microbatches"] == 2
    assert result["regularizer_to_ppo_norm_ratio"] < 0.001000001
    assert result["optimizer_projection_residual"] == 0


def test_worker_audit_rejects_fabricated_lambda():
    payload = worker_payload()
    payload["gradients"]["lambda"] *= 2

    with pytest.raises(ValueError, match="lambda formula"):
        validate_worker_evidence(
            payload,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
        )


def test_worker_audit_rejects_direct_feature_replay_drift():
    payload = worker_payload()
    replay = payload["passes"]["sphere"]
    layer = replay["layers"][TARGET]
    layer["feature_difference_l2_norm"] = 0.01
    layer["feature_relative_l2_difference"] = 1e-4
    feature_replay = replay["feature_replay"]
    feature_replay["difference_l2_norm"] = 0.01
    feature_replay["relative_l2_difference"] = 1e-4

    with pytest.raises(ValueError, match="feature drift exceeds tolerance"):
        validate_worker_evidence(
            payload,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
        )


def test_worker_audit_rejects_a_third_combined_backward_pass():
    payload = worker_payload()
    payload["passes"]["combined_forward_backward"] = 1

    with pytest.raises(ValueError, match="combined_forward_backward"):
        validate_worker_evidence(
            payload,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
        )


def test_worker_audit_rejects_gradient_assembly_inventory_drift():
    payload = worker_payload()
    payload["gradient_assembly"]["parameter_tensors"] = 4

    with pytest.raises(ValueError, match="assembly inventory"):
        validate_worker_evidence(
            payload,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
        )


def test_worker_audit_rejects_a_ppo_only_optimizer_gradient():
    payload = worker_payload()
    ppo_norm = payload["gradients"]["ppo"]["l2_norm"]
    payload["gradients"]["optimizer_assembled"]["l2_norm"] = ppo_norm
    payload["gradients"]["optimizer_reported_preclip_grad_norm"] = ppo_norm
    assembly = payload["gradient_assembly"]
    assembly["postcast_expected_l2_norm"] = ppo_norm
    assembly["actual_l2_norm"] = ppo_norm
    assembly["postcast_delta_from_ppo_l2_norm"] = 0.0
    assembly["postcast_delta_from_ppo_max_abs"] = 0.0
    assembly["postcast_values_different_from_ppo"] = 0

    with pytest.raises(ValueError, match="postcast SPHERE contribution"):
        validate_worker_evidence(
            payload,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["gradients"]["optimizer_assembled"].__setitem__(
                "l2_norm", 987654.0
            ),
            "full-tensor actual/reported assembled gradient norm",
        ),
        (
            lambda payload: payload["gradients"]["sphere"].__setitem__(
                "tensors_with_grad", 1
            ),
            "gradient coverage drifted",
        ),
        (
            lambda payload: payload["gradients"].__setitem__(
                "regularizer_contribution_l2_norm", 999.0
            ),
            "regularizer contribution norm",
        ),
        (
            lambda payload: payload["gradient_assembly"].__setitem__(
                "destination_dtype_counts", {"not-a-torch-dtype": 3}
            ),
            "dtype is unsupported",
        ),
        (
            lambda payload: payload["passes"].__setitem__("combined", {}),
            "pass inventory drifted",
        ),
    ],
)
def test_worker_audit_rejects_internally_inconsistent_evidence(mutate, message):
    payload = worker_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        validate_worker_evidence(
            payload,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
        )


def test_training_audit_derives_step_and_binds_real_artifacts(tmp_path):
    run = _make_training_run(tmp_path)
    evidence, inventory, completion = audit_training_run(
        run_dir=run,
        expected_variant="sphere_dense_last_adaptive_v1",
        registry_path=REGISTRY,
        expected_trajectories=4,
        expected_groups=2,
        rollouts_per_group=2,
    )

    assert evidence["mixed_reward_group_count"] == 1
    assert evidence["optimizer_updates_actual"] == 1
    assert inventory["global_step"] == 1
    assert completion["global_step"] == 1
    assert completion["artifacts"]["parameter_evidence_sha256"] == sha256_file(
        run / "parameter-evidence.json"
    )


def test_training_audit_rejects_coexisting_failed_worker_evidence(tmp_path):
    run = _make_training_run(tmp_path)
    _write_json(run / "sphere/worker-evidence.failed.json", {"status": "failed"})

    with pytest.raises(ValueError, match="coexisting failed"):
        audit_training_run(
            run_dir=run,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
            expected_trajectories=4,
            expected_groups=2,
            rollouts_per_group=2,
        )


def test_training_audit_rejects_worker_nonce_drift(tmp_path):
    run = _make_training_run(tmp_path)
    payload = json.loads((run / "sphere/worker-evidence.json").read_text())
    payload["run"]["nonce"] = "b" * 64
    _write_json(run / "sphere/worker-evidence.json", payload)

    with pytest.raises(ValueError, match="run nonce drifted"):
        audit_training_run(
            run_dir=run,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
            expected_trajectories=4,
            expected_groups=2,
            rollouts_per_group=2,
        )


def test_training_audit_rejects_checkpoint_step_two(tmp_path):
    run = _make_training_run(tmp_path)
    (run / "checkpoints/global_step_2").mkdir(parents=True)

    with pytest.raises(ValueError, match="exactly global_step_1"):
        audit_training_run(
            run_dir=run,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
            expected_trajectories=4,
            expected_groups=2,
            rollouts_per_group=2,
        )


def test_training_audit_rejects_compile_flag_drift(tmp_path):
    run = _make_training_run(tmp_path)
    config = run / "contract/resolved-config.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "      use_torch_compile: false", "      use_torch_compile: true"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="configuration drifted"):
        audit_training_run(
            run_dir=run,
            expected_variant="sphere_dense_last_adaptive_v1",
            registry_path=REGISTRY,
            expected_trajectories=4,
            expected_groups=2,
            rollouts_per_group=2,
        )


def test_final_audit_requires_training_receipt_and_95_percent(tmp_path):
    run = _make_training_run(tmp_path)
    audit_training_run(
        run_dir=run,
        expected_variant="sphere_dense_last_adaptive_v1",
        registry_path=REGISTRY,
        expected_trajectories=4,
        expected_groups=2,
        rollouts_per_group=2,
    )
    _write_json(run / "evaluation/before-test100.json", _evaluation(100))
    _write_json(run / "evaluation/after-test100.json", _evaluation(95))

    result = audit_final_run(
        run_dir=run, expected_variant="sphere_dense_last_adaptive_v1"
    )
    assert result["after_successes"] == 95
    assert result["meets_at_least_95_percent"] is True

    _write_json(run / "evaluation/after-test100.json", _evaluation(94))
    with pytest.raises(ValueError, match="threshold failed"):
        audit_final_run(run_dir=run, expected_variant="sphere_dense_last_adaptive_v1")


def test_final_audit_rejects_training_artifact_tampering(tmp_path):
    run = _make_training_run(tmp_path)
    audit_training_run(
        run_dir=run,
        expected_variant="sphere_dense_last_adaptive_v1",
        registry_path=REGISTRY,
        expected_trajectories=4,
        expected_groups=2,
        rollouts_per_group=2,
    )
    _write_json(run / "evaluation/before-test100.json", _evaluation(100))
    _write_json(run / "evaluation/after-test100.json", _evaluation(100))
    (run / "parameter-evidence.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact binding drifted"):
        audit_final_run(run_dir=run, expected_variant="sphere_dense_last_adaptive_v1")
