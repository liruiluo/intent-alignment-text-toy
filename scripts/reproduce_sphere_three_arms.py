#!/usr/bin/env python3
"""Reproduce the three frozen one-update dense-SPHERE experiment arms.

This is the end-to-end, single-GPU entry point used for project handoff.  It
runs the three arms sequentially.  For every arm it performs:

    frozen baseline receipt -> config resolution -> one optimizer update
    -> parameter/training audit -> frozen test100 -> final audit

The script never submits or releases an accelerator allocation.  Run it inside
an already-provisioned, otherwise idle one-GPU environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

EXPECTED_SOURCE_TAG = "sphere-one-step-repro-v1"
EXPECTED_VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"
EXPECTED_TRAIN_PARQUET_SHA256 = (
    "4ab5bfa4f55f073617795c403a10a1b45310d136691ea5dff0377a58cb991ae1"
)
EXPECTED_TEST_PARQUET_SHA256 = (
    "455a4c17c370bf40802a282215f69c71a5d21326ad185375211099bd6fe68730"
)
EXPECTED_TRAIN_JSONL_SHA256 = (
    "c7632a967e4e6ed14ebae3d87ad1f43ecab2f0da66c70a3ea11659746688a164"
)
EXPECTED_TEST_JSONL_SHA256 = (
    "b2f7560c4466949136bec09bd2e8532a27def45696c5eeed3b4fcc99de2c1675"
)
EXPECTED_BASELINE_SHA256 = (
    "f3b5b104315ba1b1098d895218779fa5a246ed9100d15eb6ac0d6de637cd968f"
)
EXPECTED_MODEL_HASHES = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "model.safetensors.index.json": (
        "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
    ),
    "model.safetensors-00001-of-00002.safetensors": (
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"
    ),
    "model.safetensors-00002-of-00002.safetensors": (
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
    ),
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "merges.txt": (
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
    ),
    "tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "vocab.json": (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    ),
}
EXPECTED_RUNTIME = {
    "python": "3.12.12",
    "torch": "2.11.0+cu130",
    "transformers": "5.5.3",
    "transfer_queue": "0.1.8",
    "sglang": "0.5.12",
    "device": "NVIDIA B300 SXM6 AC",
}
VARIANTS = (
    "sphere_dense_last_adaptive_v1",
    "sphere_dense_all_hidden_adaptive_v1",
    "sphere_dense_last_fixed_alpha_v1",
)
STAGES = (
    "reuse frozen base test100 receipt",
    "resolve the exact veRL configuration",
    "train exactly one optimizer/global update",
    "audit SPHERE gradients, checkpoint, and parameter change",
    "evaluate the trained global_step_1 checkpoint on frozen test100",
    "fail closed unless the final result is 100/100",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_sha256_sidecar(path: Path) -> None:
    path.with_suffix(".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="utf-8"
    )


def run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> None:
    print("+", " ".join(str(item) for item in command), flush=True)
    if stdout_path is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_target = subprocess.STDOUT if stderr_path is None else stderr_path.open(
        "w", encoding="utf-8"
    )
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle:
            subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=True,
                stdout=stdout_handle,
                stderr=stderr_target,
                text=True,
            )
    finally:
        if stderr_path is not None:
            stderr_target.close()


def run_logged(
    command: Sequence[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> None:
    print("+", " ".join(str(item) for item in command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 drifted: expected {expected}, got {actual}: {path}"
        )


def ensure_outside_repo(path: Path, repo_root: Path) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError:
        return
    raise ValueError(f"--run-root must be outside the source repository: {path}")


def make_runtime_env(
    *, repo_root: Path, verl_root: Path, python_bin: Path, cuda_home: Path, gpu: str,
    extra_pythonpath: str | None,
) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [str(repo_root / "src"), str(verl_root)]
    if extra_pythonpath:
        pythonpath.append(extra_pythonpath)
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "HYDRA_FULL_ERROR": "1",
            "PATH": f"{python_bin.parent}:{cuda_home / 'bin'}:{env.get('PATH', '')}",
        }
    )
    version = subprocess.check_output(
        [
            str(python_bin),
            "-c",
            "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        env=env,
        text=True,
    ).strip()
    python_prefix = Path(
        subprocess.check_output(
            [str(python_bin), "-c", "import sys; print(sys.prefix)"],
            env=env,
            text=True,
        ).strip()
    )
    runtime_cuda = python_prefix / "lib" / version / "site-packages/nvidia/cu13/lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in (
            str(cuda_home / "lib64"),
            str(runtime_cuda),
            env.get("LD_LIBRARY_PATH", ""),
        )
        if value
    )
    return env


def probe_runtime(python_bin: Path, env: dict[str, str]) -> dict[str, Any]:
    snippet = r'''
import importlib.metadata
import json
import platform
import torch
import transformers
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "transfer_queue": importlib.metadata.version("TransferQueue"),
    "sglang": importlib.metadata.version("sglang"),
    "device": torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else None,
    "gpu_count": torch.cuda.device_count(),
}, sort_keys=True))
'''
    output = subprocess.check_output(
        [str(python_bin), "-c", snippet], env=env, text=True
    )
    runtime = json.loads(output.strip().splitlines()[-1])
    expected = {**EXPECTED_RUNTIME, "gpu_count": 1}
    if runtime != expected:
        raise RuntimeError(f"runtime identity drifted: expected={expected!r}, actual={runtime!r}")
    return runtime


def find_runtime_conflicts() -> list[str]:
    """Return existing training-service processes without matching this script."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    )
    patterns = (
        "verl.trainer.main_ppo",
        "intent_alignment_toy.sphere_main",
        "raylet",
        "gcs_server",
        "sglang.launch_server",
        "torchrun",
    )
    conflicts = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        if any(pattern in command for pattern in patterns):
            conflicts.append(line.strip())
    return conflicts


def validate_source_checkout(repo_root: Path) -> str:
    if git_output(repo_root, "status", "--porcelain"):
        raise RuntimeError("source repository must be clean before reproduction")
    source_commit = git_output(repo_root, "rev-parse", "HEAD")
    release_tags = set(
        git_output(repo_root, "tag", "--points-at", "HEAD").splitlines()
    )
    if EXPECTED_SOURCE_TAG not in release_tags:
        raise RuntimeError(
            "source checkout is not the frozen reproduction release; run "
            f"`git checkout {EXPECTED_SOURCE_TAG}` first"
        )
    tag_type = git_output(
        repo_root, "cat-file", "-t", f"refs/tags/{EXPECTED_SOURCE_TAG}"
    )
    if tag_type != "tag":
        raise RuntimeError(f"{EXPECTED_SOURCE_TAG} must be an annotated Git tag")
    return source_commit


def preflight(
    *,
    repo_root: Path,
    verl_root: Path,
    model_path: Path,
    train_parquet: Path,
    test_parquet: Path,
    baseline_eval: Path,
    run_root: Path,
    python_bin: Path,
    cuda_home: Path,
    min_free_gib: int,
    env: dict[str, str],
) -> tuple[str, str, dict[str, Any]]:
    ensure_outside_repo(run_root, repo_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(
            f"--run-root must be absent or empty; refusing to mix runs: {run_root}"
        )
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python runtime is missing: {python_bin}")
    if not (cuda_home / "bin/nvcc").is_file():
        raise FileNotFoundError(f"CUDA nvcc is missing: {cuda_home / 'bin/nvcc'}")
    source_commit = validate_source_checkout(repo_root)
    verl_commit = git_output(verl_root, "rev-parse", "HEAD")
    if verl_commit != EXPECTED_VERL_COMMIT:
        raise RuntimeError(
            f"veRL commit drifted: expected {EXPECTED_VERL_COMMIT}, got {verl_commit}"
        )
    if git_output(verl_root, "status", "--porcelain"):
        raise RuntimeError("veRL checkout must be clean")

    require_hash(
        repo_root / "data/intent_alignment_train_400.jsonl",
        EXPECTED_TRAIN_JSONL_SHA256,
        "train JSONL",
    )
    require_hash(
        repo_root / "data/intent_alignment_test_100.jsonl",
        EXPECTED_TEST_JSONL_SHA256,
        "test JSONL",
    )
    require_hash(train_parquet, EXPECTED_TRAIN_PARQUET_SHA256, "train parquet")
    require_hash(test_parquet, EXPECTED_TEST_PARQUET_SHA256, "test parquet")
    require_hash(baseline_eval, EXPECTED_BASELINE_SHA256, "frozen baseline evaluation")
    for name, expected in EXPECTED_MODEL_HASHES.items():
        require_hash(model_path / name, expected, f"base model {name}")

    conflicts = find_runtime_conflicts()
    if conflicts:
        raise RuntimeError(
            "existing Ray/SGLang/veRL runtime detected; use an otherwise idle "
            "single-GPU environment:\n" + "\n".join(conflicts)
        )

    run_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"run filesystem has only {free_gib:.1f} GiB free; require >= {min_free_gib} GiB"
        )
    runtime = probe_runtime(python_bin, env)
    index_listing = subprocess.check_output(
        ["git", "-C", str(repo_root), "ls-files", "-s", "-z"]
    )
    source_index_sha256 = hashlib.sha256(index_listing).hexdigest()
    return source_commit, source_index_sha256, runtime


def make_run_contract(
    *,
    run_dir: Path,
    variant: str,
    repo_root: Path,
    source_commit: str,
    source_index_sha256: str,
    verl_root: Path,
    model_path: Path,
    train_parquet: Path,
    test_parquet: Path,
    baseline_eval: Path,
    python_bin: Path,
    cuda_home: Path,
    runtime: dict[str, Any],
    launcher: Path,
) -> dict[str, Any]:
    nonce_path = run_dir / "sphere/run-nonce"
    nonce_path.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    os.chmod(nonce_path, 0o600)
    return {
        "schema": "intent_alignment_dense_sphere_run_contract_v3",
        "created_at_utc": utc_now(),
        "run_id": run_dir.name,
        "variant": variant,
        "run_nonce_sha256": sha256_file(nonce_path),
        "objective": (
            "Qwen3.5-4B dense SPHERE, 400 train tasks x 2 rollouts, "
            "exactly one optimizer update, then frozen test100"
        ),
        "constraints": {
            "no_intermediate_evaluation": True,
            "restore_holder_after_run": True,
            "do_not_release_allocation": True,
            "no_new_allocation": True,
        },
        "algorithm": {
            "base": "GRPO",
            "sphere_variant": variant,
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
            "backward_passes": {"ppo": 1, "sphere": 1, "combined": 0},
            "optimizer_gradient_assembly": "explicit_ppo_plus_lambda_sphere",
            "full_tensor_postcast_assembly_check": True,
        },
        "source": {
            "id": f"git-{source_commit[:20]}",
            "path": str(repo_root),
            "release_tag": EXPECTED_SOURCE_TAG,
            "manifest_sha256": source_index_sha256,
            "base_commit": source_commit,
        },
        "runtime": {
            "verl_path": str(verl_root),
            "verl_commit": EXPECTED_VERL_COMMIT,
            "python": str(python_bin),
            "torch": runtime["torch"],
            "transformers": runtime["transformers"],
            "sglang": runtime["sglang"],
            "transfer_queue": runtime["transfer_queue"],
            "cuda_home": str(cuda_home),
        },
        "model": {
            "path": str(model_path),
            "config_sha256": EXPECTED_MODEL_HASHES["config.json"],
            "index_sha256": EXPECTED_MODEL_HASHES["model.safetensors.index.json"],
            "weight_shas": [
                EXPECTED_MODEL_HASHES[
                    "model.safetensors-00001-of-00002.safetensors"
                ],
                EXPECTED_MODEL_HASHES[
                    "model.safetensors-00002-of-00002.safetensors"
                ],
            ],
            "tokenizer_shas": {
                name: EXPECTED_MODEL_HASHES[name]
                for name in (
                    "chat_template.jinja",
                    "merges.txt",
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "vocab.json",
                )
            },
        },
        "data": {
            "path": str(train_parquet.parent),
            "train_jsonl_sha256": EXPECTED_TRAIN_JSONL_SHA256,
            "test_jsonl_sha256": EXPECTED_TEST_JSONL_SHA256,
            "train_parquet_sha256": EXPECTED_TRAIN_PARQUET_SHA256,
            "test_parquet_sha256": EXPECTED_TEST_PARQUET_SHA256,
        },
        "hardware": {
            "platform_task": os.environ.get("IATOY_PLATFORM_TASK", "external-existing-allocation"),
            "pod": os.environ.get("IATOY_POD", "external-single-gpu-runner"),
            "gpu_count": 1,
            "gpu": runtime["device"],
            "new_allocation_requested": False,
            "allocation_must_be_retained": True,
        },
        "evaluation": {
            "test_generation_only_after_training_endpoint": True,
            "baseline_reused_not_regenerated": True,
            "baseline_sha256": sha256_file(baseline_eval),
            "test_count": 100,
            "success_threshold": 0.95,
            "decoding": "greedy",
            "max_new_tokens": 8,
        },
        "launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
    }


def stop_owned_runtime(python_bin: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [str(python_bin), "-m", "ray", "stop", "--force"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def run_one_arm(
    *,
    variant: str,
    run_dir: Path,
    repo_root: Path,
    verl_root: Path,
    model_path: Path,
    train_parquet: Path,
    test_parquet: Path,
    baseline_eval: Path,
    python_bin: Path,
    cuda_home: Path,
    source_commit: str,
    source_index_sha256: str,
    runtime: dict[str, Any],
    base_env: dict[str, str],
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    for relative in ("contract", "evaluation", "logs", "rollouts", "sphere"):
        (run_dir / relative).mkdir(parents=True, exist_ok=False)

    arm_env = dict(base_env)
    arm_env.update(
        {
            "VERL_ROOT": str(verl_root),
            "MODEL_PATH": str(model_path),
            "TRAIN_PARQUET": str(train_parquet),
            "TEST_PARQUET": str(test_parquet),
            "RUN_DIR": str(run_dir),
            "PYTHON_BIN": str(python_bin),
            "CUDA_HOME": str(cuda_home),
            "SPHERE_VARIANT": variant,
            "IATOY_PROCESS_OWNER": run_dir.name,
            "HF_HOME": str(run_dir / "runtime/hf"),
            "XDG_CACHE_HOME": str(run_dir / "runtime/xdg"),
            "TRITON_CACHE_DIR": str(run_dir / "runtime/triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(run_dir / "runtime/torchinductor"),
        }
    )
    for key in ("HF_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        Path(arm_env[key]).mkdir(parents=True, exist_ok=True)

    exit_code = 1
    try:
        baseline_target = run_dir / "evaluation/before-test100.json"
        shutil.copyfile(baseline_eval, baseline_target)
        write_json_atomic(
            run_dir / "evaluation/before-test100-provenance.json",
            {
                "schema": "intent_alignment_reused_baseline_v1",
                "source": str(baseline_eval),
                "sha256": sha256_file(baseline_eval),
                "regenerated_in_this_run": False,
            },
        )
        write_sha256_sidecar(baseline_target)

        contract = make_run_contract(
            run_dir=run_dir,
            variant=variant,
            repo_root=repo_root,
            source_commit=source_commit,
            source_index_sha256=source_index_sha256,
            verl_root=verl_root,
            model_path=model_path,
            train_parquet=train_parquet,
            test_parquet=test_parquet,
            baseline_eval=baseline_eval,
            python_bin=python_bin,
            cuda_home=cuda_home,
            runtime=runtime,
            launcher=Path(__file__).resolve(),
        )
        contract_path = run_dir / "contract/run-contract.json"
        write_json_atomic(contract_path, contract)
        write_sha256_sidecar(contract_path)

        write_json_atomic(
            run_dir / "contract/reproduction-stage-order.json",
            {
                "schema": "intent_alignment_sphere_stage_order_v1",
                "variant": variant,
                "stages": list(STAGES),
                "post_training_evaluation_only": True,
            },
        )
        (run_dir / "logs/pipeline-start.txt").write_text(
            f"pipeline_start={utc_now()}\nrun_id={run_dir.name}\nvariant={variant}\n",
            encoding="utf-8",
        )

        sphere_launcher = repo_root / "scripts/run_verl_sphere_one_step.sh"
        run_checked(
            ["bash", str(sphere_launcher), "--cfg", "job", "--resolve"],
            cwd=repo_root,
            env=arm_env,
            stdout_path=run_dir / "contract/resolved-config.yaml",
            stderr_path=run_dir / "logs/resolve-config.log",
        )
        write_sha256_sidecar(run_dir / "contract/resolved-config.yaml")

        run_logged(
            ["bash", str(sphere_launcher)],
            cwd=repo_root,
            env=arm_env,
            log_path=run_dir / "logs/training.log",
        )

        checkpoint = run_dir / "checkpoints/global_step_1"
        trained_model = checkpoint / "actor/huggingface"
        if not (checkpoint / "data.pt").is_file():
            raise RuntimeError("global_step_1/data.pt is missing")
        if not (trained_model / "config.json").is_file():
            raise RuntimeError("global_step_1 Hugging Face config is missing")
        if not list(trained_model.glob("*.safetensors")):
            raise RuntimeError("global_step_1 Hugging Face weights are missing")
        if (run_dir / "checkpoints/global_step_2").exists():
            raise RuntimeError("global_step_2 exists; the one-update contract was violated")

        parameter_path = run_dir / "parameter-evidence.json"
        run_logged(
            [
                str(python_bin),
                str(repo_root / "scripts/audit_parameter_update.py"),
                "--base-model",
                str(model_path),
                "--trained-model",
                str(trained_model),
                "--output",
                str(parameter_path),
            ],
            cwd=repo_root,
            env=arm_env,
            log_path=run_dir / "logs/parameter-audit.log",
        )
        write_sha256_sidecar(parameter_path)

        run_logged(
            [
                str(python_bin),
                str(repo_root / "scripts/audit_sphere_training.py"),
                "training",
                "--run-dir",
                str(run_dir),
                "--variant",
                variant,
                "--registry",
                str(repo_root / "configs/sphere_variants.json"),
                "--expected-trajectories",
                "800",
                "--expected-groups",
                "400",
                "--rollouts-per-group",
                "2",
            ],
            cwd=repo_root,
            env=arm_env,
            log_path=run_dir / "logs/training-audit.log",
        )
        for name in (
            "checkpoint-inventory.json",
            "training-evidence.json",
            "training-complete.json",
        ):
            write_sha256_sidecar(run_dir / name)

        completion_path = run_dir / "training-complete.json"
        if not completion_path.is_file():
            raise RuntimeError("training audit did not create training-complete.json")

        # This is deliberately after the fail-closed training completion gate.
        after_path = run_dir / "evaluation/after-test100.json"
        run_logged(
            [
                str(python_bin),
                str(repo_root / "scripts/run_qwen.py"),
                "--model",
                str(trained_model),
                "--dataset",
                str(repo_root / "data/intent_alignment_test_100.jsonl"),
                "--prompt-contract",
                "v2-agent-loop",
                "--batch-size",
                "100",
                "--max-new-tokens",
                "8",
                "--output",
                str(after_path),
            ],
            cwd=repo_root,
            env=arm_env,
            log_path=run_dir / "logs/after-eval.log",
        )
        write_sha256_sidecar(after_path)

        run_logged(
            [
                str(python_bin),
                str(repo_root / "scripts/audit_sphere_training.py"),
                "final",
                "--run-dir",
                str(run_dir),
                "--variant",
                variant,
                "--success-threshold",
                "95",
            ],
            cwd=repo_root,
            env=arm_env,
            log_path=run_dir / "logs/final-audit.log",
        )
        status_path = run_dir / "result-status.json"
        write_sha256_sidecar(status_path)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        validate_arm_status(status, variant)
        exit_code = 0
        return status
    finally:
        stop_owned_runtime(python_bin, arm_env)
        write_json_atomic(
            run_dir / "logs/pipeline-exit.json",
            {
                "run_id": run_dir.name,
                "variant": variant,
                "exit_code": exit_code,
                "finished_at_utc": utc_now(),
            },
        )


def validate_arm_status(status: dict[str, Any], expected_variant: str) -> None:
    if status.get("variant") != expected_variant:
        raise RuntimeError(
            f"{expected_variant}: final result variant drifted: {status.get('variant')!r}"
        )
    if status.get("global_step") != 1 or status.get("optimizer_updates_actual") != 1:
        raise RuntimeError(f"{expected_variant}: optimizer/global update count is not one")
    if status.get("after_successes") != 100 or status.get("after_total") != 100:
        raise RuntimeError(
            f"{expected_variant}: reproduction did not match 100/100: "
            f"{status.get('after_successes')}/{status.get('after_total')}"
        )


def build_reproduction_summary(
    *,
    statuses: Sequence[dict[str, Any]],
    source_commit: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    actual_variants = [status.get("variant") for status in statuses]
    if actual_variants != list(VARIANTS):
        raise RuntimeError(
            "three-arm result set is incomplete or out of order: "
            f"expected={list(VARIANTS)!r}, actual={actual_variants!r}"
        )
    for variant, status in zip(VARIANTS, statuses, strict=True):
        validate_arm_status(status, variant)
    return {
        "schema": "intent_alignment_sphere_three_arm_reproduction_v1",
        "completed_at_utc": utc_now(),
        "status": "complete",
        "source_commit": source_commit,
        "source_release_tag": EXPECTED_SOURCE_TAG,
        "verl_commit": EXPECTED_VERL_COMMIT,
        "gpu": runtime["device"],
        "execution": "sequential_single_gpu",
        "arms": [
            {
                "variant": status["variant"],
                "run_id": status["run_id"],
                "optimizer_updates_actual": status["optimizer_updates_actual"],
                "test_successes": status["after_successes"],
                "test_total": status["after_total"],
                "meets_reported_100_percent": True,
            }
            for status in statuses
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run all three frozen dense-SPHERE arms sequentially on one GPU."
    )
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate all frozen inputs/runtime/GPU without launching training",
    )
    parser.add_argument("--verl-root", default=os.environ.get("VERL_ROOT"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--run-root", default=os.environ.get("RUN_ROOT"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN"))
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME"))
    parser.add_argument(
        "--train-parquet",
        default=os.environ.get(
            "TRAIN_PARQUET", str(repo_root / "data/verl-v0.9.0/train.parquet")
        ),
    )
    parser.add_argument(
        "--test-parquet",
        default=os.environ.get(
            "TEST_PARQUET", str(repo_root / "data/verl-v0.9.0/test.parquet")
        ),
    )
    parser.add_argument(
        "--baseline-eval",
        default=os.environ.get(
            "BASELINE_EVAL",
            str(
                repo_root
                / "results/qwen3.5-4b-verl-one-step-v2/before-test100.json"
            ),
        ),
    )
    parser.add_argument("--gpu", default=os.environ.get("IATOY_GPU", "0"))
    parser.add_argument("--run-tag", default=os.environ.get("RUN_TAG"))
    parser.add_argument(
        "--extra-pythonpath", default=os.environ.get("IATOY_EXTRA_PYTHONPATH")
    )
    parser.add_argument("--min-free-gib", type=int, default=256)
    return parser


def print_plan() -> None:
    print(
        json.dumps(
            {
                "schema": "intent_alignment_sphere_reproduction_plan_v1",
                "gpu_count": 1,
                "execution": "sequential",
                "variants": list(VARIANTS),
                "stages_per_variant": list(STAGES),
                "expected_optimizer_updates_each": 1,
                "expected_test_successes_each": 100,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    def stop_on_signal(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    args = build_parser().parse_args()
    if args.print_plan:
        print_plan()
        return
    missing = [
        flag
        for flag, value in (
            ("--verl-root/VERL_ROOT", args.verl_root),
            ("--model-path/MODEL_PATH", args.model_path),
            ("--run-root/RUN_ROOT", args.run_root),
            ("--python-bin/PYTHON_BIN", args.python_bin),
            ("--cuda-home/CUDA_HOME", args.cuda_home),
        )
        if not value
    ]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))
    if not re.fullmatch(r"[0-9]+", args.gpu):
        raise SystemExit("--gpu must name exactly one numeric CUDA device")

    repo_root = Path(__file__).resolve().parents[1]
    verl_root = Path(args.verl_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    python_bin = Path(args.python_bin).expanduser().resolve()
    cuda_home = Path(args.cuda_home).expanduser().resolve()
    train_parquet = Path(args.train_parquet).expanduser().resolve()
    test_parquet = Path(args.test_parquet).expanduser().resolve()
    baseline_eval = Path(args.baseline_eval).expanduser().resolve()
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_tag):
        raise SystemExit("--run-tag may contain only letters, digits, '.', '_' and '-'")

    env = make_runtime_env(
        repo_root=repo_root,
        verl_root=verl_root,
        python_bin=python_bin,
        cuda_home=cuda_home,
        gpu=args.gpu,
        extra_pythonpath=args.extra_pythonpath,
    )
    source_commit, source_index_sha256, runtime = preflight(
        repo_root=repo_root,
        verl_root=verl_root,
        model_path=model_path,
        train_parquet=train_parquet,
        test_parquet=test_parquet,
        baseline_eval=baseline_eval,
        run_root=run_root,
        python_bin=python_bin,
        cuda_home=cuda_home,
        min_free_gib=args.min_free_gib,
        env=env,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_OK",
                    "source_commit": source_commit,
                    "source_release_tag": EXPECTED_SOURCE_TAG,
                    "verl_commit": EXPECTED_VERL_COMMIT,
                    "runtime": runtime,
                    "train_parquet_sha256": EXPECTED_TRAIN_PARQUET_SHA256,
                    "test_parquet_sha256": EXPECTED_TEST_PARQUET_SHA256,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    plan_path = run_root / "reproduction-plan.json"
    write_json_atomic(
        plan_path,
        {
            "schema": "intent_alignment_sphere_reproduction_plan_v1",
            "created_at_utc": utc_now(),
            "source_commit": source_commit,
            "verl_commit": EXPECTED_VERL_COMMIT,
            "model_config_sha256": EXPECTED_MODEL_HASHES["config.json"],
            "train_parquet_sha256": EXPECTED_TRAIN_PARQUET_SHA256,
            "test_parquet_sha256": EXPECTED_TEST_PARQUET_SHA256,
            "gpu": runtime["device"],
            "gpu_count": 1,
            "execution": "sequential",
            "variants": list(VARIANTS),
            "stages_per_variant": list(STAGES),
        },
    )

    statuses = []
    for index, variant in enumerate(VARIANTS, start=1):
        run_id = f"iatoy-{variant}-qwen35-4b-one-step-{run_tag}"
        run_dir = run_root / run_id
        print(f"\n[{index}/{len(VARIANTS)}] START {variant} -> {run_dir}", flush=True)
        status = run_one_arm(
            variant=variant,
            run_dir=run_dir,
            repo_root=repo_root,
            verl_root=verl_root,
            model_path=model_path,
            train_parquet=train_parquet,
            test_parquet=test_parquet,
            baseline_eval=baseline_eval,
            python_bin=python_bin,
            cuda_home=cuda_home,
            source_commit=source_commit,
            source_index_sha256=source_index_sha256,
            runtime=runtime,
            base_env=env,
        )
        statuses.append(status)
        print(
            f"[{index}/{len(VARIANTS)}] PASS {variant}: "
            f"update={status['optimizer_updates_actual']}, "
            f"test={status['after_successes']}/{status['after_total']}",
            flush=True,
        )

    summary = build_reproduction_summary(
        statuses=statuses,
        source_commit=source_commit,
        runtime=runtime,
    )
    summary_path = run_root / "reproduction-summary.json"
    write_json_atomic(summary_path, summary)
    write_sha256_sidecar(summary_path)
    print(f"\nREPRODUCTION_COMPLETE summary={summary_path}")
    for arm in summary["arms"]:
        print(
            f"{arm['variant']}: update={arm['optimizer_updates_actual']} "
            f"test={arm['test_successes']}/{arm['test_total']}"
        )


if __name__ == "__main__":
    main()
