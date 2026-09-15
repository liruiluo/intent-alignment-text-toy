import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "qwen3.5-4b-verl-one-step-v2"
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_published_results", ROOT / "scripts" / "verify_published_results.py"
)
VERIFY_MODULE = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(VERIFY_MODULE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((RESULT_DIR / name).read_text(encoding="utf-8"))


def test_publication_manifest_binds_source_and_published_artifacts():
    manifest = load("artifact-manifest.json")
    assert manifest["schema"] == "intent_alignment_public_result_manifest_v2"
    assert manifest["source_commit"] == (
        "100f0f4c117779f941cf38f61ef095f58c2ea1a6"
    )
    expected_source_subset = {
        "after-test100.json": (
            "8c6a6a1255bcaea72133f0a137c0839bd52160ed188a9505c40136adbc80149c"
        ),
        "before-test100.json": (
            "0104c0385a9da5ff5e6d551bb31ebaf84a63b71b2ba16af29390240f4216237a"
        ),
        "paired-comparison.json": (
            "5fe927aaec0b372c1a84bb499a5ebc3b4efbb5a58ca59311a8909177d9d28156"
        ),
        "parameter-evidence.json": (
            "e2c2f57a95723cda4000e8a5504c895440338a201f0444e80273d425c5e04db3"
        ),
        "result-status.json": (
            "809f28f0fc9e5121bb66b6316e4b7cbbf1e347a69993415bf23faa7f176309c7"
        ),
        "rollouts-step1.jsonl": (
            "49a326573bfb51c22cb75a1dee0efdfea1a8db9881cbd2e4d3b579c76a740a44"
        ),
        "training-complete.json": (
            "e8febeadadd320d7f278a0ae4b1267cd374ec6c49076fed027ef852d5da0af14"
        ),
        "training-evidence.json": (
            "5995459319a1c48c62c5aa9ce698f3602f78cd9c37334e5084c60bd015645444"
        ),
        "training-log.redacted.txt": (
            "8dc44833b60ad727a054115c7857499c1dd8d1f5958d30d3bcd207ec85ba5ec6"
        ),
    }
    assert manifest["source_artifact_sha256"] | expected_source_subset == (
        manifest["source_artifact_sha256"]
    )
    for name, expected in expected_source_subset.items():
        assert manifest["source_artifact_sha256"][name] == expected
    for name, expected in manifest["published_artifact_sha256"].items():
        assert sha256_file(RESULT_DIR / name) == expected


def test_formal_result_status_and_metrics_are_consistent():
    status = load("result-status.json")
    before = load("before-test100.json")
    after = load("after-test100.json")
    paired = load("paired-comparison.json")
    training = load("training-evidence.json")
    parameter = load("parameter-evidence.json")

    assert status["status"] == "complete"
    assert status["global_step"] == status["optimizer_steps_actual"] == 1
    assert status["optimizer_steps_expected"] == 1
    assert status["before_successes"] == before["summary"]["successes"] == 100
    assert status["after_successes"] == after["summary"]["successes"] == 100
    assert status["success_rate_delta"] == paired["success_rate_delta"] == 0.0
    assert status["mixed_reward_group_count"] == training[
        "mixed_reward_group_count"
    ] == 66
    assert status["weights_changed"] is parameter["weights_changed"] is True
    assert status["effective_bfloat16_weights_changed"] is True
    assert status["hardware"]["reused_existing_allocation"] is True
    assert status["hardware"]["new_allocation_requested"] is False


def test_public_result_artifacts_do_not_expose_machine_local_identifiers():
    forbidden = (
        "/home/",
        "/Users/",
        "/dev/shm",
        "6.235.",
        "luolirui-1-",
        "master-0",
    )
    for path in RESULT_DIR.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{path.name} contains private marker {marker!r}"
        for match in re.findall(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])", text):
            assert match == "127.0.0.1", f"{path.name} contains IP {match!r}"


def test_full_publication_verifier_passes():
    summary = VERIFY_MODULE.verify(RESULT_DIR)
    assert summary == {
        "status": "PASS",
        "published_artifacts": 13,
        "source_artifacts": 13,
        "before_successes": 100,
        "after_successes": 100,
        "optimizer_updates": 1,
        "trajectories": 800,
        "mixed_reward_groups": 66,
        "changed_probes": 6,
        "effective_bfloat16_changed_values": 8,
    }


def test_redacted_launcher_remains_valid_bash():
    subprocess.run(
        ["bash", "-n", str(RESULT_DIR / "launch.redacted.sh")],
        check=True,
        capture_output=True,
        text=True,
    )
