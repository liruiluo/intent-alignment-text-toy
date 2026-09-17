import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "showcase" / "intent-alignment-showcase.html"
RESULT_DIR = ROOT / "results" / "qwen3.5-4b-verl-one-step-v2"
BEFORE = RESULT_DIR / "before-test100.json"
AFTER = RESULT_DIR / "after-test100.json"
PAIRED = RESULT_DIR / "paired-comparison.json"
TRAINING = RESULT_DIR / "training-evidence.json"
PARAMETER = RESULT_DIR / "parameter-evidence.json"
MANIFEST = RESULT_DIR / "artifact-manifest.json"


def embedded_evidence() -> dict:
    html = SHOWCASE.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="evidenceData" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match, "standalone showcase must embed its evidence payload"
    return json.loads(match.group(1))


def test_showcase_embeds_exact_frozen_v2_results():
    showcase = embedded_evidence()
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    after = json.loads(AFTER.read_text(encoding="utf-8"))
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    parameter = json.loads(PARAMETER.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert showcase["protocol"]["name"] == after["protocol"]["name"]
    assert showcase["protocol"]["dataset_sha256"] == after["protocol"][
        "dataset_sha256"
    ]
    assert showcase["protocol"]["decoding"] == after["protocol"]["decoding"]
    assert showcase["protocol"]["metric"] == after["protocol"]["metric"]
    assert showcase["protocol"]["threshold"] == after["protocol"]["threshold"]
    assert showcase["protocol"]["source_commit"] == manifest["source_commit"]
    assert showcase["protocol"]["source_before_result_sha256"] == manifest[
        "source_artifact_sha256"
    ]["before-test100.json"]
    assert showcase["protocol"]["source_after_result_sha256"] == manifest[
        "source_artifact_sha256"
    ]["after-test100.json"]
    assert showcase["protocol"]["source_paired_comparison_sha256"] == manifest[
        "source_artifact_sha256"
    ]["paired-comparison.json"]
    assert showcase["protocol"]["published_before_result_sha256"] == manifest[
        "published_artifact_sha256"
    ]["before-test100.json"]
    assert showcase["protocol"]["published_after_result_sha256"] == manifest[
        "published_artifact_sha256"
    ]["after-test100.json"]
    assert showcase["protocol"]["published_paired_comparison_sha256"] == manifest[
        "published_artifact_sha256"
    ]["paired-comparison.json"]

    assert showcase["dataset"] == {"train": 400, "test": 100, "max_steps": 5}
    assert showcase["before"] == before["summary"]
    assert showcase["after"] == after["summary"]
    assert showcase["tasks"] == after["tasks"]
    assert showcase["paired"] == {
        key: len(value) for key, value in paired["paired"].items()
    }

    assert showcase["training"] == {
        "framework": "veRL v0.9.0",
        "algorithm": "GRPO",
        "optimizer_updates": training["optimizer_updates_expected"],
        "global_step": training["global_step"],
        "trajectories": training["trajectory_count"],
        "groups": training["grpo_group_count"],
        "rollouts_per_group": training["rollouts_per_group"],
        "mixed_reward_groups": training["mixed_reward_group_count"],
        "actor_grad_norm": training["positive_finite_actor_grad_norm"][0],
    }
    assert showcase["parameter"] == {
        "policy_core_tensor_count": parameter["policy_core_tensor_count"],
        "changed_probe_count": parameter["changed_probe_count"],
        "effective_bfloat16_changed_value_count": parameter[
            "effective_bfloat16_changed_value_count"
        ],
        "weights_changed": parameter["weights_changed"],
        "effective_bfloat16_weights_changed": parameter[
            "effective_bfloat16_weights_changed"
        ],
    }


def test_showcase_contains_real_three_step_agent_trace():
    evidence = embedded_evidence()
    tasks = {task["id"]: task for task in evidence["tasks"]}
    task = tasks["test-right-3-f03"]

    assert task["instruction"] == "由原点出发，限5步，抵达右侧3米的位置。"
    assert task["actions"] == ["RIGHT", "RIGHT", "RIGHT"]
    assert task["trajectory"] == [[0, 0], [1, 0], [2, 0], [3, 0]]
    assert task["success"] is True


def test_showcase_is_a_single_private_path_free_offline_html():
    html = SHOWCASE.read_text(encoding="utf-8")

    assert "<script src=" not in html
    assert re.search(r'<link[^>]+rel="stylesheet"', html) is None
    assert "http://" not in html and "https://" not in html
    assert "/home/" not in html and "/Users/" not in html and "D:\\" not in html
    assert "/dev/shm" not in html
    assert "6.235." not in html
    assert "luolirui-1-" not in html
    assert "enable_thinking = false" in html
    assert "HTML 不在浏览器中运行模型" in html
    assert 'if (tasks.length !== 100)' in html
    assert 'tasks.filter(task => task.success).length !== 100' in html

SPHERE_RESULT_DIR = ROOT / "results" / "qwen3.5-4b-sphere-one-step-v1"


def embedded_sphere_evidence() -> dict:
    html = SHOWCASE.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="sphereEvidenceData" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match, "standalone showcase must embed its SPHERE evidence payload"
    return json.loads(match.group(1))


def test_showcase_embeds_exact_three_arm_sphere_summary():
    embedded = embedded_sphere_evidence()
    summary = json.loads(
        (SPHERE_RESULT_DIR / "summary.json").read_text(encoding="utf-8")
    )
    assert embedded == summary
    assert embedded["all_arms_passed"] is True
    assert len(embedded["arms"]) == 3
    assert all(arm["optimizer_updates_actual"] == 1 for arm in embedded["arms"])
    assert all(arm["after_successes"] == 100 for arm in embedded["arms"])


def test_showcase_surfaces_three_sphere_arms_for_windows_demo():
    html = SHOWCASE.read_text(encoding="utf-8")
    assert "3 / 3 达到 ≥95%" in html
    assert "All hidden expert layers" in html
    assert "每臂恰好 1 update" in html
    assert "SPHERE 三臂摘要 SHA" in html
