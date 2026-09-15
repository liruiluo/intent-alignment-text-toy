import importlib.util
from pathlib import Path

from intent_alignment_toy.evidence import EXPECTED_CONFIG_NORMALIZATION, MISSING

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_parameter_update", ROOT / "scripts" / "audit_parameter_update.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_json_diff_matches_frozen_qwen_export_normalization():
    before = {
        "transformers_version": "4.57.0.dev0",
        "text_config": {"mtp_num_hidden_layers": 1},
        "vision_config": {},
    }
    after = {
        "bos_token_id": None,
        "dtype": "bfloat16",
        "eos_token_id": 248046,
        "pad_token_id": 248044,
        "transformers_version": "5.5.3",
        "text_config": {
            "bos_token_id": None,
            "mtp_num_hidden_layers": 0,
            "pad_token_id": None,
            "partial_rotary_factor": 0.25,
        },
        "vision_config": {"dtype": "bfloat16"},
    }

    assert MODULE.json_diff(before, after) == EXPECTED_CONFIG_NORMALIZATION


def test_json_diff_records_removed_keys_and_type_changes():
    assert MODULE.json_diff({"a": 1, "b": []}, {"a": "1"}) == {
        "/a": {"before": 1, "after": "1"},
        "/b": {"before": [], "after": MISSING},
    }


def test_tensor_classification_keeps_policy_core_separate():
    assert MODULE.is_policy_core_tensor(
        "model.language_model.layers.8.input_layernorm.weight"
    )
    assert not MODULE.is_policy_core_tensor("mtp.layers.0.weight")
    assert not MODULE.is_policy_core_tensor("model.visual.encoder.weight")
    assert MODULE.is_visual_tensor("model.visual.encoder.weight")


def test_weight_file_manifest_records_bytes_hash_and_shard_id(tmp_path):
    first = tmp_path / "model.safetensors-00001-of-00002.safetensors"
    second = tmp_path / "model.safetensors"
    first.write_bytes(b"base")
    second.write_bytes(b"trained")

    rows = MODULE.weight_file_manifest(tmp_path, [first.name, second.name])

    assert rows[0]["bytes"] == 4
    assert rows[0]["shard_id"] == "00001-of-00002"
    assert rows[0]["sha256"] == MODULE.sha256_file(first)
    assert rows[1]["bytes"] == 7
    assert rows[1]["shard_id"] == "model.safetensors"
    assert rows[1]["sha256"] == MODULE.sha256_file(second)
