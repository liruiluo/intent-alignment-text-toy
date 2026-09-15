from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_verl_one_step.sh"


def test_ray_tmpdir_is_short_and_independent_of_long_run_path():
    script = LAUNCHER.read_text(encoding="utf-8")

    assert 'export RAY_TMPDIR="/tmp/iatoy-ray-${run_dir_sha256:0:12}"' in script
    assert 'RAY_TMPDIR="$RUN_DIR/runtime-cache/ray"' not in script

    # Ray 2.51 appends this shape before creating its plasma-store socket.
    longest_expected = (
        "/tmp/iatoy-ray-"
        + "f" * 12
        + "/ray/session_2026-09-15_23-34-16_088688_9999999/sockets/plasma_store"
    )
    assert len(longest_expected.encode()) <= 107


def test_training_launcher_requires_pinned_sglang_runtime():
    script = LAUNCHER.read_text(encoding="utf-8")

    assert '"transfer_queue": "0.1.8"' in script
    assert 'importlib.metadata.version("TransferQueue")' in script
    assert '"sglang": "0.5.12"' in script
    assert "actor_rollout_ref.rollout.name=sglang" in script
    assert "actor_rollout_ref.rollout.name=vllm" not in script
    assert "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton" in script
    assert "+actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch" in script
    assert "actor_rollout_ref.rollout.load_format=safetensors" in script
    assert "TVM_FFI_CACHE_DIR" in script
    assert "CUDA_HOME" in script
    assert "site-packages/nvidia/cu13/lib" in script
    assert 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$runtime_cuda_lib' in script
    assert "4ab5bfa4f55f073617795c403a10a1b45310d136691ea5dff0377a58cb991ae1" in script
    assert "455a4c17c370bf40802a282215f69c71a5d21326ad185375211099bd6fe68730" in script
