#!/usr/bin/env bash
set -Eeuo pipefail

: "${VERL_ROOT:?set VERL_ROOT to the exact veRL checkout}"
: "${MODEL_PATH:?set MODEL_PATH to Qwen3.5-4B checkpoint}"
: "${TRAIN_PARQUET:?set TRAIN_PARQUET}"
: "${TEST_PARQUET:?set TEST_PARQUET}"
: "${RUN_DIR:?set RUN_DIR}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AGENT_LOOP_CONFIG="$REPO_ROOT/configs/intent_alignment_agent_loop.yaml"
PYTHON_BIN=${PYTHON_BIN:-python3}
TRAIN_MODULE=${TRAIN_MODULE:-verl.trainer.main_ppo}
ACTOR_GRADIENT_CHECKPOINTING=${ACTOR_GRADIENT_CHECKPOINTING:-true}
ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-true}
readonly TRAIN_BATCH_SIZE=400
readonly ROLLOUT_N=2
readonly PPO_MINI_BATCH_SIZE=400
readonly PPO_MICRO_BATCH_SIZE=4
readonly LOGPROB_MICRO_BATCH_SIZE=8
readonly ACTOR_LR=1e-6
readonly EXPECTED_VERL_COMMIT=483b8a009ba3a97563edee3a19887e4862b8094a
readonly EXPECTED_TRAIN_PARQUET_SHA256=4ab5bfa4f55f073617795c403a10a1b45310d136691ea5dff0377a58cb991ae1
readonly EXPECTED_TEST_PARQUET_SHA256=455a4c17c370bf40802a282215f69c71a5d21326ad185375211099bd6fe68730
readonly EXPECTED_MODEL_CONFIG_SHA256=ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670
readonly EXPECTED_MODEL_INDEX_SHA256=cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d
readonly EXPECTED_MODEL_WEIGHT_1_SHA256=26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61
readonly EXPECTED_MODEL_WEIGHT_2_SHA256=cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188

mkdir -p "$RUN_DIR" "$RUN_DIR/rollouts"
[[ -f "$VERL_ROOT/verl/trainer/main_ppo.py" ]]
case "$TRAIN_MODULE" in
  verl.trainer.main_ppo|intent_alignment_toy.sphere_main) ;;
  *) echo "unsupported TRAIN_MODULE: $TRAIN_MODULE" >&2; exit 64 ;;
esac
case "$ACTOR_GRADIENT_CHECKPOINTING:$ACTOR_USE_TORCH_COMPILE" in
  true:true|true:false|false:true|false:false) ;;
  *) echo "actor boolean overrides must be true/false" >&2; exit 64 ;;
esac
[[ -f "$MODEL_PATH/config.json" ]]
[[ -f "$TRAIN_PARQUET" ]]
[[ -f "$TEST_PARQUET" ]]
[[ -f "$AGENT_LOOP_CONFIG" ]]

if (( $# > 0 )); then
  if (( $# != 3 )) || [[ "$1" != "--cfg" || "$2" != "job" || "$3" != "--resolve" ]]; then
    echo "Only the exact diagnostic arguments '--cfg job --resolve' are allowed" >&2
    exit 64
  fi
fi

check_sha256() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA256 drifted: expected $expected, got $actual" >&2
    exit 65
  fi
}

actual_verl_commit=$(git -c safe.directory="$VERL_ROOT" -C "$VERL_ROOT" rev-parse HEAD)
[[ "$actual_verl_commit" == "$EXPECTED_VERL_COMMIT" ]] || {
  echo "veRL commit drifted: expected $EXPECTED_VERL_COMMIT, got $actual_verl_commit" >&2
  exit 65
}
[[ -z "$(git -c safe.directory="$VERL_ROOT" -C "$VERL_ROOT" status --porcelain)" ]] || {
  echo "veRL checkout must be clean" >&2
  exit 65
}
check_sha256 "$TRAIN_PARQUET" "$EXPECTED_TRAIN_PARQUET_SHA256" "train parquet"
check_sha256 "$TEST_PARQUET" "$EXPECTED_TEST_PARQUET_SHA256" "test parquet"
check_sha256 "$MODEL_PATH/config.json" "$EXPECTED_MODEL_CONFIG_SHA256" "model config"
check_sha256 "$MODEL_PATH/model.safetensors.index.json" "$EXPECTED_MODEL_INDEX_SHA256" "model index"
"$PYTHON_BIN" - "$MODEL_PATH/model.safetensors.index.json" <<'PY_MODEL_INDEX'
import json
import sys

expected = {
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
}
with open(sys.argv[1], encoding="utf-8") as handle:
    actual = set(json.load(handle)["weight_map"].values())
if actual != expected:
    raise SystemExit(f"model index shard set drifted: expected={sorted(expected)!r}, actual={sorted(actual)!r}")
PY_MODEL_INDEX
check_sha256 "$MODEL_PATH/model.safetensors-00001-of-00002.safetensors" "$EXPECTED_MODEL_WEIGHT_1_SHA256" "model shard 1"
check_sha256 "$MODEL_PATH/model.safetensors-00002-of-00002.safetensors" "$EXPECTED_MODEL_WEIGHT_2_SHA256" "model shard 2"

: "${CUDA_HOME:?set CUDA_HOME to a CUDA 13 toolkit that supports sm_103}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || {
  echo "CUDA_HOME does not provide nvcc: $CUDA_HOME" >&2
  exit 65
}
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
python_prefix=$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')
python_minor=$("$PYTHON_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
python_header="$python_prefix/include/$python_minor/Python.h"
[[ -f "$python_header" ]] || {
  echo "Python runtime header required by Triton is missing: $python_header" >&2
  exit 65
}
runtime_cuda_lib="$python_prefix/lib/$python_minor/site-packages/nvidia/cu13/lib"
[[ -d "$runtime_cuda_lib" ]] || {
  echo "Python runtime does not provide CUDA 13 libraries: $runtime_cuda_lib" >&2
  exit 65
}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$runtime_cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 13\.' || {
  echo "Qwen3.5 SGLang JIT requires CUDA 13.x on B300" >&2
  exit 65
}

"$PYTHON_BIN" - <<'PY_RUNTIME_GATE'
import importlib.metadata
import platform

import torch
import transformers

expected = {
    "python": "3.12.12",
    "torch": "2.11.0+cu130",
    "transformers": "5.5.3",
    "transfer_queue": "0.1.8",
    "sglang": "0.5.12",
    "device": "NVIDIA B300 SXM6 AC",
}
actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "transfer_queue": importlib.metadata.version("TransferQueue"),
    "sglang": importlib.metadata.version("sglang"),
    "device": torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else None,
}
if actual != expected:
    raise SystemExit(f"runtime identity drifted: expected={expected!r}, actual={actual!r}")
print(f"runtime identity verified: {actual}")
PY_RUNTIME_GATE

if (( TRAIN_BATCH_SIZE != PPO_MINI_BATCH_SIZE )); then
  # veRL multiplies actor.ppo_mini_batch_size by rollout.n internally.
  echo "This one-step contract requires PPO_MINI_BATCH_SIZE == TRAIN_BATCH_SIZE" >&2
  exit 64
fi

export PYTHONPATH="$REPO_ROOT/src:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

# Ray appends a roughly 67-byte session/socket suffix. Keep its local root short
# so the final AF_UNIX socket path stays below Linux's 107-byte limit even when
# RUN_DIR is a long shared-storage experiment path.
run_dir_sha256=$(printf '%s' "$RUN_DIR" | sha256sum | awk '{print $1}')
ray_tmp_base=${IATOY_RAY_TMP_BASE:-/tmp}
[[ -d "$ray_tmp_base" && -w "$ray_tmp_base" ]] || {
  echo "Ray temp base must be an existing writable directory: $ray_tmp_base" >&2
  exit 65
}
export RAY_TMPDIR="$ray_tmp_base/iatoy-ray-${run_dir_sha256:0:12}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$RAY_TMPDIR/tvm-ffi}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$RAY_TMPDIR/torch-extensions}"
mkdir -p "$RAY_TMPDIR" "$TVM_FFI_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

"$PYTHON_BIN" -m "$TRAIN_MODULE" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN_PARQUET" \
  data.val_files="$TEST_PARQUET" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length=512 \
  data.max_response_length=384 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.return_raw_chat=true \
  ++data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing="$ACTOR_GRADIENT_CHECKPOINTING" \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile="$ACTOR_USE_TORCH_COMPILE" \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="$ACTOR_USE_TORCH_COMPILE" \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.load_format=safetensors \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.agent.num_workers=64 \
  actor_rollout_ref.rollout.agent.default_agent_loop=intent_alignment_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=intent_alignment_text_toy \
  trainer.experiment_name="$(basename "$RUN_DIR")" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  "$@"
