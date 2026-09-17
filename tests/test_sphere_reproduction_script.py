import hashlib
import importlib.util
import json
import subprocess

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_sphere_three_arms.py"
README = ROOT / "README.md"

EXPECTED_VARIANTS = [
    "sphere_dense_last_adaptive_v1",
    "sphere_dense_all_hidden_adaptive_v1",
    "sphere_dense_last_fixed_alpha_v1",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_arm_reproduction_plan_is_complete_and_ordered():
    result = subprocess.run(
        [str(SCRIPT), "--print-plan"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["variants"] == EXPECTED_VARIANTS
    assert plan["execution"] == "sequential"
    assert plan["gpu_count"] == 1
    assert plan["expected_optimizer_updates_each"] == 1
    assert plan["expected_test_successes_each"] == 100
    stages = plan["stages_per_variant"]
    assert stages.index("train exactly one optimizer/global update") < stages.index(
        "evaluate the trained global_step_1 checkpoint on frozen test100"
    )


def test_reproducer_calls_real_train_eval_and_audit_entrypoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "run_verl_sphere_one_step.sh" in source
    assert "audit_parameter_update.py" in source
    assert '"training"' in source
    assert "run_qwen.py" in source
    assert '"final"' in source
    assert 'checkpoints/global_step_1' in source
    assert 'checkpoints/global_step_2' in source
    assert 'default=256' in source
    assert "after_successes" in source
    assert "new_allocation_requested" in source
    assert "9Nctl" not in source


def test_committed_parquet_inputs_match_training_launcher_contract():
    assert sha256(ROOT / "data/verl-v0.9.0/train.parquet") == (
        "4ab5bfa4f55f073617795c403a10a1b45310d136691ea5dff0377a58cb991ae1"
    )
    assert sha256(ROOT / "data/verl-v0.9.0/test.parquet") == (
        "455a4c17c370bf40802a282215f69c71a5d21326ad185375211099bd6fe68730"
    )


def test_readme_documents_one_command_three_arm_reproduction():
    readme = README.read_text(encoding="utf-8")
    assert "scripts/reproduce_sphere_three_arms.py" in readme
    assert "训练 → 训练审计 → test100 → 最终审计" in readme
    assert "reproduction-summary.json" in readme
    assert all(variant in readme for variant in EXPECTED_VARIANTS)


def load_reproducer():
    spec = importlib.util.spec_from_file_location("sphere_reproducer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def completed_status(variant: str) -> dict:
    return {
        "variant": variant,
        "run_id": f"run-{variant}",
        "global_step": 1,
        "optimizer_updates_actual": 1,
        "after_successes": 100,
        "after_total": 100,
    }


def test_release_and_tokenizer_inputs_are_frozen():
    module = load_reproducer()
    assert module.EXPECTED_SOURCE_TAG == "sphere-one-step-repro-v2"
    assert {
        "chat_template.jinja",
        "merges.txt",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    } <= set(module.EXPECTED_MODEL_HASHES)


def test_source_checkout_requires_clean_annotated_release_tag(monkeypatch):
    module = load_reproducer()
    answers = {
        ("status", "--porcelain"): "",
        ("rev-parse", "HEAD"): "abc123",
        ("tag", "--points-at", "HEAD"): module.EXPECTED_SOURCE_TAG,
        ("cat-file", "-t", f"refs/tags/{module.EXPECTED_SOURCE_TAG}"): "tag",
    }
    monkeypatch.setattr(module, "git_output", lambda _root, *args: answers[args])
    assert module.validate_source_checkout(ROOT) == "abc123"

    answers[("tag", "--points-at", "HEAD")] = ""
    with pytest.raises(RuntimeError, match="frozen reproduction release"):
        module.validate_source_checkout(ROOT)


def test_summary_requires_all_three_exact_one_step_100_of_100_arms():
    module = load_reproducer()
    statuses = [completed_status(variant) for variant in EXPECTED_VARIANTS]
    summary = module.build_reproduction_summary(
        statuses=statuses,
        source_commit="abc123",
        runtime={"device": "NVIDIA B300 SXM6 AC"},
    )
    assert [arm["variant"] for arm in summary["arms"]] == EXPECTED_VARIANTS
    assert all(arm["optimizer_updates_actual"] == 1 for arm in summary["arms"])
    assert all(arm["test_successes"] == 100 for arm in summary["arms"])

    invalid_cases = [
        statuses[:2],
        [*statuses[:2], {**statuses[2], "optimizer_updates_actual": 2}],
        [*statuses[:2], {**statuses[2], "after_successes": 99}],
    ]
    for invalid in invalid_cases:
        with pytest.raises(RuntimeError):
            module.build_reproduction_summary(
                statuses=invalid,
                source_commit="abc123",
                runtime={"device": "NVIDIA B300 SXM6 AC"},
            )
