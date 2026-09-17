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

export IATOY_SPHERE_VARIANT="$SPHERE_VARIANT"
export IATOY_SPHERE_CONFIG_PATH="$SPHERE_CONFIG_PATH"
export IATOY_SPHERE_EVIDENCE_PATH="${SPHERE_EVIDENCE_PATH:-$RUN_DIR/sphere/worker-evidence.json}"
export TRAIN_MODULE=intent_alignment_toy.sphere_main
export ACTOR_GRADIENT_CHECKPOINTING=false
export ACTOR_USE_TORCH_COMPILE=false

mkdir -p "$(dirname "$IATOY_SPHERE_EVIDENCE_PATH")"
exec bash "$REPO_ROOT/scripts/run_verl_one_step.sh" "$@"
