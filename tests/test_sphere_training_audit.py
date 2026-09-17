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
    combined_projection = ppo_projection + coefficient * sphere_projection

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
    }
    replay = {
        "microbatches": 2,
        "identity_matches": 2,
        "penalty_sum": 8.0,
        "collection_surrogate_dot_sum": 12.0,
        "raw_surrogate_sum": 12.0,
        "raw_surrogate_matches_collection": True,
        "surrogate_sum": 8.0,
        "surrogate_matches_penalty": True,
        "layers": {TARGET: replay_layer},
    }
    checks = {
        "logical_minibatch_penalty_finite_nonzero": True,
        "logical_feature_gradients_finite_nonzero": True,
        "sphere_gradient_finite_nonzero": True,
        "lambda_formula_matches": True,
        "ratio_formula_matches": True,
        "sphere_replay_matches_logical_penalty": True,
        "combined_replay_matches_logical_penalty": True,
        "sphere_replay_features_match_collection": True,
        "combined_replay_features_match_collection": True,
        "all_replay_microbatch_identities_match": True,
        "combined_projection_matches_sum": True,
        "optimizer_preclip_grad_norm_finite_nonzero": True,
        "all_target_activation_gradients_nonzero": True,
        "optimizer_step_exactly_one": True,
    }
    return {
        "schema": "intent_alignment_dense_sphere_worker_evidence_v2",
        "status": "passed",
        "variant": "sphere_dense_last_adaptive_v1",
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
            "gradient_checkpointing_required_off": True,
            "torch_compile_required_off": True,
        },
        "registry": {"sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest()},
        "runtime": {"world_size": 1, "rank": 0},
        "targets": {"count": 1, "names": [TARGET]},
        "passes": {
            "ppo_only_forward_backward": 1,
            "sphere_only_forward_backward": 1,
            "combined_forward_backward": 1,
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
            "combined": replay,
        },
        "gradients": {
            "ppo": gradient(ppo_norm, ppo_projection),
            "sphere": gradient(sphere_norm, sphere_projection),
            "combined": gradient(2.1, combined_projection),
            "lambda": coefficient,
            "expected_lambda": coefficient,
            "regularizer_contribution_l2_norm": coefficient * sphere_norm,
            "regularizer_to_ppo_norm_ratio": ratio,
            "expected_regularizer_to_ppo_norm_ratio": ratio,
            "predicted_combined_projection": combined_projection,
            "actual_combined_projection": combined_projection,
            "combined_projection_residual": 0.0,
            "combined_projection_tolerance": 5e-4,
            "optimizer_reported_preclip_grad_norm": 2.1,
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
data:
  train_batch_size: 400
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: false
  actor:
    ppo_mini_batch_size: 400
    ppo_micro_batch_size_per_gpu: 4
    ppo_epochs: 1
    use_torch_compile: false
    fsdp_config:
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
    _write_json(run / "sphere/worker-evidence.json", worker_payload())
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
        "schema": "intent_alignment_dense_sphere_run_contract_v2",
        "run_id": run.name,
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
    assert result["combined_projection_residual"] == 0


def test_worker_audit_rejects_fabricated_lambda():
    payload = worker_payload()
    payload["gradients"]["lambda"] *= 2

    with pytest.raises(ValueError, match="lambda formula"):
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
