#!/usr/bin/env bash
# run.sh — Variant + render driver for the VLM-unreliability audit harness.
#
# For every scene directory under RESULTS_DIR that contains a scene-state file,
# generate content variants then render the requested audit phase(s).
#
# Usage:
#   ./run.sh                 # phase 'all' over results/*
#   ./run.sh 1a              # only phase 1a
#   RESULTS_DIR=mydir ./run.sh 2
set -euo pipefail

PYTHON="${PYTHON:-/Users/anson/miniforge3/envs/vlmunr/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SEED="${SEED:-42}"
PHASE="${1:-all}"
# Optional: export HSSD_DIR=/path/to/hssd-models  (needed for stk-only scenes)

shopt -s nullglob
found=0
for scene_dir in "$RESULTS_DIR"/*/; do
  if [[ -f "$scene_dir/hsm_scene_state.json" || -f "$scene_dir/stk_scene_state.json" ]]; then
    found=1
    echo "[run] === $scene_dir ==="
    echo "[run] variants (seed=$SEED)"
    "$PYTHON" vlmunr_variants.py --scene-dir "$scene_dir" --seed "$SEED"
    echo "[run] render (phase=$PHASE)"
    "$PYTHON" vlmunr_render.py --scene-dir "$scene_dir" --phase "$PHASE"
  fi
done

if [[ "$found" -eq 0 ]]; then
  echo "[run] no scene-state files found under $RESULTS_DIR/*/" >&2
  exit 1
fi
echo "[run] done"
