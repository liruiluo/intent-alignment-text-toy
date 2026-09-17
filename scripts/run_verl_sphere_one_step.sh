#!/usr/bin/env bash
set -Eeuo pipefail

: "${SPHERE_VARIANT:?set SPHERE_VARIANT to one frozen variant id}"
: "${RUN_DIR:?set RUN_DIR}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly SPHERE_CONFIG_PATH="$REPO_ROOT/configs/sphere_variants.json"
PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" - "$SPHERE_CONFIG_PATH" "$SPHERE_VARIANT" <<'PY_VARIANT_GATE'
import sys
from intent_alignment_toy.sphere_contract import load_sphere_variant

variant = load_sphere_variant(sys.argv[1], sys.argv[2])
print(
    "SPHERE variant verified: "
    f"{variant.name}, layers={variant.layer_mode}, "
    f"target_scale={variant.target_scale_mode}, fixed_alpha={variant.fixed_alpha}, "
    f"rho={variant.rho}"
)
PY_VARIANT_GATE

readonly SPHERE_DIR="$RUN_DIR/sphere"
readonly EXPECTED_EVIDENCE_PATH="$SPHERE_DIR/worker-evidence.json"
readonly FAILED_EVIDENCE_PATH="$SPHERE_DIR/worker-evidence.failed.json"
readonly RUN_NONCE_PATH="$SPHERE_DIR/run-nonce"

if [[ -n "${SPHERE_EVIDENCE_PATH:-}" && "$SPHERE_EVIDENCE_PATH" != "$EXPECTED_EVIDENCE_PATH" ]]; then
  echo "SPHERE_EVIDENCE_PATH must equal $EXPECTED_EVIDENCE_PATH" >&2
  exit 72
fi
mkdir -p "$SPHERE_DIR"
if [[ -e "$EXPECTED_EVIDENCE_PATH" || -e "$FAILED_EVIDENCE_PATH" ]]; then
  echo "refusing to reuse an existing SPHERE worker evidence target" >&2
  exit 73
fi
if [[ ! -e "$RUN_NONCE_PATH" ]]; then
  "$PYTHON_BIN" - "$RUN_NONCE_PATH" <<'PY_NONCE'
import secrets
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(secrets.token_hex(32) + "\n")
except FileExistsError:
    pass
PY_NONCE
fi
IATOY_SPHERE_RUN_NONCE=$(tr -d '\r\n' < "$RUN_NONCE_PATH")
[[ "$IATOY_SPHERE_RUN_NONCE" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SPHERE run nonce" >&2; exit 74; }

export IATOY_SPHERE_VARIANT="$SPHERE_VARIANT"
export IATOY_SPHERE_CONFIG_PATH="$SPHERE_CONFIG_PATH"
export IATOY_SPHERE_EVIDENCE_PATH="$EXPECTED_EVIDENCE_PATH"
export IATOY_SPHERE_RUN_ID="$(basename "$RUN_DIR")"
export IATOY_SPHERE_RUN_NONCE
export TRAIN_MODULE=intent_alignment_toy.sphere_main
export ACTOR_GRADIENT_CHECKPOINTING=false
export ACTOR_USE_TORCH_COMPILE=false

exec bash "$REPO_ROOT/scripts/run_verl_one_step.sh" "$@"
