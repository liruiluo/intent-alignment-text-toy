import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "qwen3.5-4b-verl-one-step-v2"
SPEC = importlib.util.spec_from_file_location(
    "compare_results", ROOT / "scripts" / "compare_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ARTIFACT_NAMES = {
    "before": "before-test100.json",
    "after": "after-test100.json",
    "parameter": "parameter-evidence.json",
    "training": "training-evidence.json",
    "rollout": "rollouts-step1.jsonl",
    "training_log": "training-log.redacted.txt",
    "checkpoint": "checkpoint-inventory.json",
    "run_contract": "run-contract.redacted.json",
    "resolved_config": "resolved-config.redacted.yaml",
    "launcher": "launch.redacted.sh",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_bundle(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, name in ARTIFACT_NAMES.items():
        source = RESULT_DIR / name
        target = tmp_path / name
        shutil.copyfile(source, target)
        paths[key] = target
    return paths


def build(paths: dict[str, Path]) -> dict:
    return MODULE.build_report(
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


def mutate_json(path: Path, mutator) -> None:
    payload = load_json(path)
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_strict_comparison_accepts_published_verl_run(tmp_path):
    paths = stage_bundle(tmp_path)
    report = build(paths)

    assert report["parameter_evidence"]["weights_changed"]
    assert report["parameter_evidence"][
        "effective_bfloat16_changed_value_count"
    ] == 8
    assert report["training_evidence"]["trajectory_count"] == 800
    assert len(report["paired"]["unchanged_success"]) == 100


def test_comparison_rejects_decoding_drift(tmp_path):
    paths = stage_bundle(tmp_path)
    mutate_json(
        paths["after"],
        lambda payload: payload["protocol"]["decoding"].__setitem__(
            "max_new_tokens", 1
        ),
    )
    with pytest.raises(ValueError, match="decoding drifted"):
        build(paths)


def test_comparison_rejects_task_identity_drift(tmp_path):
    paths = stage_bundle(tmp_path)
    mutate_json(
        paths["after"],
        lambda payload: payload["tasks"][0].__setitem__(
            "instruction", payload["tasks"][0]["instruction"] + " changed"
        ),
    )
    with pytest.raises(ValueError, match="instruction drifted"):
        build(paths)


def test_comparison_rejects_fabricated_successful_trajectory(tmp_path):
    paths = stage_bundle(tmp_path)

    def fabricate(payload: dict) -> None:
        row = payload["tasks"][0]
        row.update(
            actions=[],
            raw_outputs=[],
            trajectory=[[0, 0]],
            position=[0, 0],
            steps_used=0,
            remaining_steps=row["max_steps"],
            reason="running",
            terminated=False,
            success=True,
        )

    mutate_json(paths["after"], fabricate)
    with pytest.raises(ValueError, match="trajectory is not terminal"):
        build(paths)


def test_comparison_rejects_empty_or_mutated_weight_manifest(tmp_path):
    paths = stage_bundle(tmp_path)
    mutate_json(
        paths["after"],
        lambda payload: payload["model"]["weights"].__setitem__("files", []),
    )
    with pytest.raises(ValueError, match="weight manifest is empty"):
        build(paths)

    paths = stage_bundle(tmp_path / "mutated-hash")
    mutate_json(
        paths["after"],
        lambda payload: payload["model"]["weights"]["files"][0].__setitem__(
            "sha256", "0" * 64
        ),
    )
    with pytest.raises(ValueError, match="evaluated weight binding drifted"):
        build(paths)


def test_comparison_rejects_unbound_or_fabricated_parameter_evidence(tmp_path):
    paths = stage_bundle(tmp_path)
    mutate_json(
        paths["parameter"],
        lambda evidence: evidence.__setitem__("after_config_sha256", "3" * 64),
    )
    with pytest.raises(ValueError, match="trained config binding drifted"):
        build(paths)

    paths = stage_bundle(tmp_path / "fabricated-probe")

    def fabricate_probe(evidence: dict) -> None:
        evidence["probes"] = [
            {
                "name": "not.a.real.tensor",
                "changed_float32_values": 1,
                "effective_bfloat16_changed_values": 1,
            }
        ]
        evidence["changed_probe_count"] = 1
        evidence["effective_bfloat16_changed_value_count"] = 1

    mutate_json(paths["parameter"], fabricate_probe)
    with pytest.raises(ValueError, match="probe identities drifted"):
        build(paths)


def test_comparison_rejects_invalid_training_evidence(tmp_path):
    paths = stage_bundle(tmp_path)
    mutate_json(
        paths["training"],
        lambda evidence: evidence.__setitem__("global_step", 2),
    )
    with pytest.raises(ValueError, match="global_step.*drifted"):
        build(paths)

    paths = stage_bundle(tmp_path / "infinite-grad")

    def infinite_grad(evidence: dict) -> None:
        evidence["positive_finite_actor_grad_norm"] = [float("inf")]
        evidence["actor_grad_norm_values"] = [float("inf")]

    mutate_json(paths["training"], infinite_grad)
    with pytest.raises(ValueError, match="invalid actor gradient"):
        build(paths)

    paths = stage_bundle(tmp_path / "mixed-count")
    mutate_json(
        paths["training"],
        lambda evidence: evidence.__setitem__("mixed_reward_group_count", 1),
    )
    with pytest.raises(ValueError, match="mixed_reward_group_count.*drifted"):
        build(paths)


def test_comparison_rejects_tampered_rollout_or_log(tmp_path):
    paths = stage_bundle(tmp_path)
    paths["rollout"].write_text(
        paths["rollout"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="rollout SHA256 drifted"):
        build(paths)

    paths = stage_bundle(tmp_path / "log")
    paths["training_log"].write_text("fake\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training log SHA256 drifted"):
        build(paths)


def test_comparison_rejects_semantic_model_identity_drift(tmp_path):
    paths = stage_bundle(tmp_path)
    mutate_json(
        paths["after"],
        lambda payload: payload["model"]["config_fingerprint"].__setitem__(
            "model_type", "different"
        ),
    )
    with pytest.raises(ValueError, match="model identity field 'model_type' drifted"):
        build(paths)


def test_comparison_rejects_tokenizer_or_runtime_drift(tmp_path):
    paths = stage_bundle(tmp_path)
    mutate_json(
        paths["after"],
        lambda payload: payload["model"]["tokenizer_fingerprint"].__setitem__(
            "vocab_size", 1
        ),
    )
    with pytest.raises(ValueError, match="tokenizer fingerprint drifted"):
        build(paths)

    paths = stage_bundle(tmp_path / "runtime")
    mutate_json(
        paths["after"],
        lambda payload: payload["runtime"].__setitem__(
            "device", "NVIDIA A100-SXM4-80GB"
        ),
    )
    with pytest.raises(ValueError, match="runtime field 'device' drifted"):
        build(paths)


def test_comparator_pins_frozen_dataset_and_full_model_hashes():
    assert MODULE.EXPECTED_DATASET_SHA256 == (
        "b2f7560c4466949136bec09bd2e8532a27def45696c5eeed3b4fcc99de2c1675"
    )
    assert MODULE.EXPECTED_BASE_MODEL_CONFIG_SHA256 == (
        "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
    )
    assert MODULE.EXPECTED_MODEL_CONFIG_FINGERPRINT["sha256"] == (
        "59a96f1b0661530c8c919d4529ece05245e106b30c2ac46e4a0f0c4be9f6cc78"
    )
    assert MODULE.sha256_file(MODULE.FROZEN_TEST) == MODULE.EXPECTED_DATASET_SHA256


def test_comparator_rejects_mutated_local_test_file(tmp_path, monkeypatch):
    drifted = tmp_path / MODULE.FROZEN_TEST.name
    drifted.write_bytes(MODULE.FROZEN_TEST.read_bytes() + b"\n")
    monkeypatch.setattr(MODULE, "FROZEN_TEST", drifted)

    with pytest.raises(ValueError, match="local frozen test100 drifted"):
        MODULE.frozen_tasks()
