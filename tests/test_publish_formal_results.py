import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_formal_results", ROOT / "scripts" / "publish_formal_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def contract() -> dict:
    return {
        "source": {"path": "/home/private/work/code/commit"},
        "data": {
            "train_parquet": "/home/private/work/data/train.parquet",
            "test_parquet": "/home/private/work/data/test.parquet",
        },
        "model": {"path": "/dev/shm/model"},
        "runtime": {
            "python": "/dev/shm/runtime/bin/python",
            "runtime_root": "/dev/shm/runtime",
            "cuda_home": "/dev/shm/cuda",
            "transfer_queue_overlay": "/home/private/work/runtime/overlay",
        },
        "hardware": {
            "platform_task": "luolirui-1-private-task",
            "pod": "luolirui-1-private-task-master-0",
            "pod_ip": "6.235.1.2",
        },
    }


def test_redaction_removes_machine_local_identifiers(tmp_path):
    run_dir = Path("/home/private/work/run")
    text = "\n".join(
        [
            str(run_dir),
            "/home/private/work/code/commit/file.py",
            "/home/private/work/data/train.parquet",
            "/dev/shm/runtime/bin/python",
            "server=6.235.1.2",
            "task=luolirui-1-private-task-master-0",
            "dashboard=http://127.0.0.1:8265   ",
        ]
    )
    redacted = MODULE.redact_machine_text(text, contract(), run_dir)
    output = tmp_path / "redacted.txt"
    output.write_text(redacted, encoding="utf-8")

    MODULE.ensure_private_markers_absent(output)
    assert "/REDACTED/RUN_DIR" in redacted
    assert "/REDACTED/SOURCE_ROOT/file.py" in redacted
    assert "/REDACTED/TRAIN_PARQUET" in redacted
    assert "/REDACTED/PYTHON_BIN" in redacted
    assert "REDACTED_POD_IP" in redacted
    assert "http://127.0.0.1:8265" in redacted
    assert "http://127.0.0.1:8265   " not in redacted


def test_public_evaluation_payload_redacts_and_binds_source(tmp_path):
    source = tmp_path / "before.json"
    source.write_text(
        '{"model":{"path":"/dev/shm/private-model"},'
        '"runtime":{"hostname":"private-master-0"}}\n',
        encoding="utf-8",
    )

    payload = MODULE.public_evaluation_payload(source, "before")

    assert payload["model"]["path"] == "Qwen3.5-4B frozen base checkpoint"
    assert payload["runtime"]["hostname"] == "single-B300-runner"
    assert payload["source_artifact_sha256"] == MODULE.sha256_file(source)
