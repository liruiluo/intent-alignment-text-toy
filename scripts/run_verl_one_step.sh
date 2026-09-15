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
readonly TRAIN_BATCH_SIZE=400
readonly ROLLOUT_N=2
readonly PPO_MINI_BATCH_SIZE=400
readonly PPO_MICRO_BATCH_SIZE=4
readonly LOGPROB_MICRO_BATCH_SIZE=8
readonly ACTOR_LR=1e-6
readonly EXPECTED_VERL_COMMIT=483b8a009ba3a97563edee3a19887e4862b8094a
readonly EXPECTED_TRAIN_PARQUET_SHA256=0ac6025c3976485be9d0ee2df27fd326515deb05d09ed127ae9c33482894594f
readonly EXPECTED_TEST_PARQUET_SHA256=8c70d231dd8b3bc13d1529e30b37c8961c9a8662efc60040a5a93d08a74ae741
readonly EXPECTED_MODEL_CONFIG_SHA256=496d39ff3964cea8009c004e5d89fbd7801526992f699ff3f2934b12cb7e8892
readonly EXPECTED_MODEL_INDEX_SHA256=7ea014c2759b3fc80f435c023ffc8b44ce91e31d109f1d48123f0e14c365b2d7
readonly EXPECTED_MODEL_WEIGHT_1_SHA256=9cb7db23bcd0858bd4f26ac49c6ee88999cb5721e73e7ac0b09acc18774c89ef
readonly EXPECTED_MODEL_WEIGHT_2_SHA256=8359a9f4e0f3a2d9fa5e2dfcf5e0de82b966e67767e74edddd29580b55089fa4

mkdir -p "$RUN_DIR" "$RUN_DIR/rollouts"
[[ -f "$VERL_ROOT/verl/trainer/main_ppo.py" ]]
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
[[ $(find "$MODEL_PATH" -maxdepth 1 -type f -name '*.safetensors' | wc -l) -eq 2 ]] || {
  echo "frozen model must contain exactly two safetensors shards" >&2
  exit 65
}
check_sha256 "$MODEL_PATH/model-00001-of-00002.safetensors" "$EXPECTED_MODEL_WEIGHT_1_SHA256" "model shard 1"
check_sha256 "$MODEL_PATH/model-00002-of-00002.safetensors" "$EXPECTED_MODEL_WEIGHT_2_SHA256" "model shard 2"

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
    "device": "NVIDIA B300 SXM6 AC",
}
actual = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "transfer_queue": importlib.metadata.version("TransferQueue"),
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
export RAY_TMPDIR="/tmp/iatoy-ray-${run_dir_sha256:0:12}"
mkdir -p "$RAY_TMPDIR"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
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
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
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
