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
