#!/usr/bin/env bash
# run.sh — Render driver for the VLM-unreliability audit harness (per-subdir).
#
# Renders every scene subfolder (one holding a scene-state file) under RUN_DIR.
# This mirrors cli.py's --path mode: it does NOT generate, only renders.
#
# Usage:
#   ./run.sh                         # --mode all (full six-factor sweep) over <RUN_DIR>/*
#   ./run.sh single                  # --mode single (baseline only)
#   ./run.sh all                     # --mode all
#   RUN_DIR=outputs/20260708-023434 ./run.sh all
#
# Equivalently, for a whole run dir at once:
#   conda activate hsm
#   python cli.py --path outputs/20260708-023434 --render-all
set -euo pipefail

PYTHON="${PYTHON:-/Users/anson/miniforge3/envs/vlmunr/bin/python}"
RUN_DIR="${RUN_DIR:-outputs}"
MODE="${1:-all}"
# Optional: export HSSD_DIR=/path/to/hssd-models  (needed for stk-only scenes)

shopt -s nullglob
found=0
# Render base/ and every variant_* subfolder that carries a state file, in order.
render_one() {
  local d="$1"
  if [[ -f "$d/hsm_scene_state.json" || -f "$d/stk_scene_state.json" ]]; then
    found=1
    echo "[run] === $d (mode=$MODE) ==="
    "$PYTHON" vlmunr_render.py --scene-dir "$d" --mode "$MODE"
  fi
}

for sub in base variant_*; do
  [[ -d "$RUN_DIR/$sub" ]] && render_one "$RUN_DIR/$sub"
done
# Also handle a flat layout (state files directly under scene dirs).
for scene_dir in "$RUN_DIR"/*/; do
  name="$(basename "$scene_dir")"
  [[ "$name" == "base" || "$name" == variant_* ]] && continue
  render_one "${scene_dir%/}"
done

if [[ "$found" -eq 0 ]]; then
  echo "[run] no renderable scene subfolders found under $RUN_DIR" >&2
  exit 1
fi
echo "[run] done"
