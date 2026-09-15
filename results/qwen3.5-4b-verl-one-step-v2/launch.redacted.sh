#!/usr/bin/env bash
set -Eeuo pipefail

readonly RUN_ID=intent-v2-verl090-sglang0512-grpo-step1-qwen35-4b-full-b300-20260916-run3
readonly COMMIT=100f0f4c117779f941cf38f61ef095f58c2ea1a6
readonly WS=/REDACTED/PRIVATE_PATH
readonly RUN="$WS/experiments/$RUN_ID"
readonly SRC="$WS/code/$COMMIT"
readonly DATA="$WS/data/$COMMIT"
readonly VERL="$WS/upstream/verl-v0.9.0"
readonly MODEL=/REDACTED/BASE_MODEL_ROOT
readonly RUNTIME=/REDACTED/RUNTIME_ROOT
readonly TQ_SITE="$WS/runtime/py312-torch211-verl090-tq018/lib/python3.12/site-packages"
readonly PYTHON="$RUNTIME/bin/python"
readonly CUDA_HOME=/REDACTED/CUDA_HOME
readonly GPU_MARKER=/tmp/crg-holder-yield
readonly CPU_MARKER=/tmp/agentmemory-formal-cpu-active
readonly MARKER_PAYLOAD="$RUN_ID $(date -Is) pid=$$"

cleanup() {
  local rc=$?
  set +e
  exec 9>/tmp/amg-holder-marker-transaction.lock
  flock -x 9
  for marker in "$GPU_MARKER" "$CPU_MARKER"; do
    if [[ -f "$marker" ]] && [[ "$(cat "$marker" 2>/dev/null)" == "$MARKER_PAYLOAD" ]]; then
      rm -f -- "$marker"
    fi
  done
  flock -u 9
  printf '{"run_id":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$RUN_ID" "$rc" "$(date -Is)" > "$RUN/logs/pipeline-exit.json"
  return "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$(cat "$SRC/.source-commit")" == "$COMMIT" ]]
(
  cd "$SRC"
  sha256sum -c "$RUN/contract/source-files.sha256"
)
sha256sum -c "$RUN/contract/INPUT_SHA256SUMS"
[[ ! -e "$RUN/evaluation/before-test100.json" ]]
[[ ! -e "$RUN/checkpoints" ]]
[[ ! -e "$RUN/evaluation/after-test100.json" ]]
[[ ! -e "$RUN/evaluation/paired-comparison.json" ]]

if pgrep -af '[v]erl[.]trainer|[r]aylet|[s]glang.*server|[v]llm.*server|[t]orchrun' > "$RUN/logs/prelaunch-conflicts.txt"; then
  cat "$RUN/logs/prelaunch-conflicts.txt" >&2
  echo "refusing to launch beside an existing formal stack" >&2
  exit 73
fi

exec 9>/tmp/amg-holder-marker-transaction.lock
flock -x 9
for marker in "$GPU_MARKER" "$CPU_MARKER"; do
  if [[ -e "$marker" ]]; then
    echo "foreign holder marker exists: $marker" >&2
    exit 74
  fi
  tmp="${marker}.${RUN_ID}.tmp"
  printf '%s\n' "$MARKER_PAYLOAD" > "$tmp"
  mv "$tmp" "$marker"
done
flock -u 9

for _ in $(seq 1 60); do
  if grep -q 'mode=yield' /tmp/crg-holder.state 2>/dev/null; then
    break
  fi
  sleep 1
done
grep -q 'mode=yield' /tmp/crg-holder.state

export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TQ_SITE:$SRC/src:$VERL"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$RUN/runtime-cache/hf"
export XDG_CACHE_HOME="$RUN/runtime-cache/xdg"
export TRITON_CACHE_DIR="$RUN/runtime-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/runtime-cache/torchinductor"
export PATH="$RUNTIME/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$RUNTIME/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

{
  echo "pipeline_start=$(date -Is)"
  echo "run_id=$RUN_ID"
  echo "source_commit=$COMMIT"
  echo "runtime=$RUNTIME"
  echo "model=$MODEL"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
} | tee "$RUN/logs/pipeline-start.txt"

cd "$SRC"
"$PYTHON" scripts/run_qwen.py \
  --model "$MODEL" \
  --dataset data/intent_alignment_test_100.jsonl \
  --prompt-contract v2-agent-loop \
  --batch-size 100 \
  --max-new-tokens 8 \
  --output "$RUN/evaluation/before-test100.json" \
  2>&1 | tee "$RUN/logs/before-eval.log"
sha256sum "$RUN/evaluation/before-test100.json" > "$RUN/evaluation/before-test100.sha256"

export VERL_ROOT="$VERL"
export MODEL_PATH="$MODEL"
export TRAIN_PARQUET="$DATA/train.parquet"
export TEST_PARQUET="$DATA/test.parquet"
export RUN_DIR="$RUN"
export PYTHON_BIN="$PYTHON"
export CUDA_HOME
bash -x scripts/run_verl_one_step.sh 2>&1 | tee "$RUN/logs/training.log"

readonly CHECKPOINT="$RUN/checkpoints/global_step_1"
readonly HF_CHECKPOINT="$CHECKPOINT/actor/huggingface"
[[ -d "$CHECKPOINT/actor" ]]
[[ -f "$CHECKPOINT/data.pt" ]]
[[ -f "$HF_CHECKPOINT/config.json" ]]
[[ -f "$HF_CHECKPOINT/model.safetensors.index.json" ]]
[[ $(find "$HF_CHECKPOINT" -maxdepth 1 -type f -name '*.safetensors' | wc -l) -gt 0 ]]

"$PYTHON" - "$RUN" <<'PY_TRAIN_AUDIT'
import collections
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

run = Path(sys.argv[1])
files = sorted((run / "rollouts").glob("*.jsonl"))
if len(files) != 1:
    raise SystemExit(f"expected exactly one rollout JSONL, got {[p.name for p in files]}")
rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 800:
    raise SystemExit(f"expected 800 trajectories, got {len(rows)}")
steps = {int(row["step"]) for row in rows}
if steps != {1}:
    raise SystemExit(f"expected only global step 1 rollout rows, got {sorted(steps)}")
groups = collections.defaultdict(list)
for row in rows:
    score = float(row["score"])
    if not math.isfinite(score):
        raise SystemExit("non-finite rollout score")
    groups[row["input"]].append(score)
if len(groups) != 400 or any(len(scores) != 2 for scores in groups.values()):
    raise SystemExit(f"expected 400 groups of two trajectories, got {len(groups)} groups")
mixed = sum(min(scores) != max(scores) for scores in groups.values())
if mixed < 1:
    raise SystemExit("no mixed-reward GRPO group; no learning signal was demonstrated")
text = (run / "logs/training.log").read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
values = [
    float(value)
    for value in re.findall(
        r"actor/grad_norm(?:['\"])?\s*[:=]\s*(?:np\.float\d*\()?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        text,
    )
]
finite_nonzero = [value for value in values if math.isfinite(value) and value > 0]
if not finite_nonzero:
    raise SystemExit(f"no positive finite actor/grad_norm found; parsed={values!r}")
payload = {
    "schema": "intent_alignment_one_step_training_evidence_v1",
    "run_id": run.name,
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "global_step": 1,
    "optimizer_updates_expected": 1,
    "trajectory_count": len(rows),
    "grpo_group_count": len(groups),
    "rollouts_per_group": 2,
    "mixed_reward_group_count": mixed,
    "reward_values": sorted({float(row["score"]) for row in rows}),
    "actor_grad_norm_values": values,
    "positive_finite_actor_grad_norm": finite_nonzero,
    "rollout_file": str(files[0]),
}
out = run / "training-evidence.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY_TRAIN_AUDIT
sha256sum "$RUN/training-evidence.json" > "$RUN/training-evidence.sha256"
printf '{"run_id":"%s","global_step":1,"trainer_exit_code":0,"completed_at":"%s"}\n' \
  "$RUN_ID" "$(date -Is)" > "$RUN/training-complete.json"
sha256sum "$RUN/training-complete.json" > "$RUN/training-complete.sha256"

# The declared training endpoint is complete above. Only now evaluate the saved checkpoint.
"$PYTHON" scripts/run_qwen.py \
  --model "$HF_CHECKPOINT" \
  --dataset data/intent_alignment_test_100.jsonl \
  --prompt-contract v2-agent-loop \
  --batch-size 100 \
  --max-new-tokens 8 \
  --output "$RUN/evaluation/after-test100.json" \
  2>&1 | tee "$RUN/logs/after-eval.log"
sha256sum "$RUN/evaluation/after-test100.json" > "$RUN/evaluation/after-test100.sha256"

"$PYTHON" scripts/compare_results.py \
  --before "$RUN/evaluation/before-test100.json" \
  --after "$RUN/evaluation/after-test100.json" \
  --output "$RUN/evaluation/paired-comparison.json" \
  2>&1 | tee "$RUN/logs/compare.log"
sha256sum "$RUN/evaluation/paired-comparison.json" > "$RUN/evaluation/paired-comparison.sha256"

"$PYTHON" - "$RUN" <<'PY_RESULT'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

run = Path(sys.argv[1])
def h(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before = json.loads((run / "evaluation/before-test100.json").read_text())
after = json.loads((run / "evaluation/after-test100.json").read_text())
paired = json.loads((run / "evaluation/paired-comparison.json").read_text())
training = json.loads((run / "training-evidence.json").read_text())
status = {
    "schema": "intent_alignment_verl_one_step_result_v1",
    "run_id": run.name,
    "status": "complete",
    "finished_at": subprocess.check_output(["date", "-Is"], text=True).strip(),
    "global_step": 1,
    "optimizer_steps_expected": 1,
    "before_successes": before["summary"]["successes"],
    "before_total": before["summary"]["total"],
    "before_success_rate": before["summary"]["success_rate"],
    "after_successes": after["summary"]["successes"],
    "after_total": after["summary"]["total"],
    "after_success_rate": after["summary"]["success_rate"],
    "success_rate_delta": paired["success_rate_delta"],
    "weights_changed": paired["parameter_evidence"]["weights_changed"],
    "mixed_reward_group_count": training["mixed_reward_group_count"],
    "positive_finite_actor_grad_norm": training["positive_finite_actor_grad_norm"],
    "artifacts": {
        "before_test100_sha256": h(run / "evaluation/before-test100.json"),
        "after_test100_sha256": h(run / "evaluation/after-test100.json"),
        "paired_comparison_sha256": h(run / "evaluation/paired-comparison.json"),
        "training_log_sha256": h(run / "logs/training.log"),
        "training_evidence_sha256": h(run / "training-evidence.json"),
        "training_complete_sha256": h(run / "training-complete.json"),
    },
}
(run / "result-status.json").write_text(
    json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(status, ensure_ascii=False, indent=2))
PY_RESULT
sha256sum "$RUN/result-status.json" > "$RUN/result-status.sha256"
echo "PIPELINE_COMPLETE $RUN_ID $(date -Is)"
