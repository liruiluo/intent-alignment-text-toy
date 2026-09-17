import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "qwen3.5-4b-sphere-one-step-v1"
SPEC = importlib.util.spec_from_file_location(
    "verify_sphere_published_results",
    ROOT / "scripts" / "verify_sphere_published_results.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_three_arm_publication_verifier_passes():
    result = MODULE.verify(RESULT_DIR)
    assert result == {
        "status": "PASS",
        "arms": 3,
        "test_successes_each": 100,
        "optimizer_updates_each": 1,
        "trajectories_each": 800,
        "mixed_reward_groups": {
            "sphere_dense_last_adaptive_v1": 61,
            "sphere_dense_all_hidden_adaptive_v1": 60,
            "sphere_dense_last_fixed_alpha_v1": 67,
        },
    }


def test_three_arm_summary_has_the_frozen_variants_and_one_update_each():
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    arms = {row["variant"]: row for row in summary["arms"]}
    assert set(arms) == {
        "sphere_dense_last_adaptive_v1",
        "sphere_dense_all_hidden_adaptive_v1",
        "sphere_dense_last_fixed_alpha_v1",
    }
    assert all(row["optimizer_updates_actual"] == 1 for row in arms.values())
    assert all(row["global_step"] == 1 for row in arms.values())
    assert all(row["after_successes"] == row["after_total"] == 100 for row in arms.values())
    assert arms["sphere_dense_all_hidden_adaptive_v1"]["target_count"] == 32
    assert arms["sphere_dense_last_fixed_alpha_v1"]["fixed_alpha"] == 1.0
