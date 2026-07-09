#!/usr/bin/env bash
# gen.sh — Generate HSM scenes for the VLM-unreliability audit harness.
#
# Loops over a list of room prompts and calls HSM's main.py for each, writing
# each scene under results/<name>/. HSM ALWAYS emits stk_scene_state.json.
#
# NOTE: hsm_scene_state.json (preferred by render.py because it carries
# mesh_path + pose) is only emitted when the scene is saved with
# save_scene_state=True. main.py does NOT expose this as a CLI flag; the
# pipeline's final scene.save(...) calls use the default (False). To emit it,
# patch the save() calls in hsm_core/scene/processing/scene_pipeline.py (or the
# default in hsm_core/scene/core/manager.py:save) to pass save_scene_state=True.
# The renderer falls back to stk_scene_state.json (which needs an HSSD dir to
# resolve meshes) when hsm_scene_state.json is absent.
#
# Usage: ./gen.sh
set -euo pipefail

PYTHON="${PYTHON:-/Users/anson/miniforge3/envs/hsm/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# name|prompt pairs
PROMPTS=(
  "bedroom_01|a cozy bedroom with a bed, nightstand, and a wardrobe"
  "living_01|a living room with a sofa, coffee table, and a bookshelf"
  "office_01|a home office with a desk, office chair, and a shelf"
)

mkdir -p "$RESULTS_DIR"

for entry in "${PROMPTS[@]}"; do
  name="${entry%%|*}"
  prompt="${entry#*|}"
  out="$RESULTS_DIR/$name"
  echo "[gen] $name :: $prompt"
  "$PYTHON" main.py -d "$prompt" --output "$out"
done

echo "[gen] done -> $RESULTS_DIR/"
