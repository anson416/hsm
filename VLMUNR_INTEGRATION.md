# VLM-Unreliability Integration Layer (HSM)

A self-contained rendering + content-variant layer added on top of HSM (Hierarchical
Scene Motifs) for the VLM-evaluator audit harness. HSM generates scene *geometry/layout*
but does **not** render RGB — this layer adds a headless Blender (bpy) renderer, a fixed
factor-sweep contract, and content-variant generators.

## Files added (all at repo root)

| File | Purpose |
|------|---------|
| `vlmunr_bpa.py` | Blender helper library (`Builder`, `Renderer`, `clear`, `initialize`, `import_obj`, `transform`). Copied verbatim from the `vlm-unreliability` repo. |
| `vlmunr_hdri/` | 8 HDRI environment maps (`*.exr`) + `license.txt`. |
| `vlmunr_config.py` | Factor levels, baseline config, `phase_levels(phase)` helper. |
| `vlmunr_render.py` | Scene parsing (PURE functions) + headless bpy renderer with phase sweeps. |
| `vlmunr_variants.py` | Removal + worst-match content-variant generators (PURE removal/renumber + lazy retrieval hook). |
| `tests/test_vlmunr_integration.py` | pytest: filename builders, removal/renumber, PURE transform/decode, bpy smoke render. |
| `gen.sh` | Scene-generation driver (loops prompts -> `main.py`). |
| `run.sh` | Variant + render driver (loops scene dirs). |

## Run commands

```bash
PY=/Users/anson/miniforge3/envs/vlmunr/bin/python

# 1. Generate scenes (writes results/<name>/)
./gen.sh

# 2. Generate variants + render all phases for every scene dir
./run.sh                # phase 'all'
./run.sh 1a             # single phase

# Or invoke directly:
$PY vlmunr_variants.py --scene-dir results/bedroom_01 --seed 42
$PY vlmunr_render.py    --scene-dir results/bedroom_01 --phase 2
# For stk-only scenes, set HSSD_DIR so meshes resolve:
HSSD_DIR=/path/to/hssd-models $PY vlmunr_render.py --scene-dir results/bedroom_01 --phase 1a

# Tests
$PY -m pytest tests/ -q
```

## Filename scheme

Written under `<scene-dir>/renderings/`:

- **Transparent master** (per resolution/focal/camera/HDRI):
  `render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png`
  e.g. `render_512_50_0_0_city.png`
- **Composited per background gray** (master alpha-composited over an RGB gray):
  `render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png`
  e.g. `render_512_50_128_128_128_0_0_city.png`

Variant scene-state files are written next to the source as
`<stem>_<variant>.json`, e.g. `hsm_scene_state_variant_half.json`.

## Factor levels (`vlmunr_config.py`)

| Factor | Levels | Baseline |
|--------|--------|----------|
| `RESOLUTIONS` | 224, 256, 384, 448, 512, 640, 768, 1024 | 512 |
| `FOCAL_LENGTHS` | 24, 35, 50, 85, 100, 200 | 50 |
| `BACKGROUND_GRAYS` | 0, 18, 65, 117, 128, 186, 204, 255 | (128,128,128) |
| `HDRIS` | city, courtyard, forest, interior, night, studio, sunrise, sunset | city |
| `PITCHES` | 0, 30, 60, 90 (0 == top-down in bpa convention) | 0 |
| `YAWS` | 0, 30, … , 330 | 0 |

`phase_levels(phase)`:
- `1a` resolution sweep · `1b` background sweep · `1c` HDRI sweep · `1d` focal sweep
- `2` pitch × yaw camera-angle grid (4 × 12 = 48 poses)

Camera framing uses `Renderer.render_perspective(out, center, radius, rotation=(pitch,0,yaw), …)`
with the bounding sphere computed from the imported meshes. Changing the HDRI re-`initialize()`s
the world.

## Content variants (`vlmunr_variants.py`)

**Removal** (pure, fully unit-tested):
- `variant_half` / `variant_quarter` / `variant_eighth` — keep `max(1, round(n/k))`
  objects (k = 2/4/8) via a seeded sample, then renumber `id`/`index` (stk) or `id`
  (hsm) contiguously from 0. Deterministic for a fixed `--seed`.

**Worst-match** (`variant_alt_0` / `variant_alt_2` / `variant_alt_4`):
- Intent: swap each object's retrieved asset for a worst-CLIP-match at ascending rank
  0/2/4. HSM retrieval ranks best-first via `(-sim).argsort()`; worst-match is the
  ascending argsort.
- **Degrades gracefully**: the retrieval hook lazily imports `hsm_core.retrieval`, but a
  real lookup needs the 72 GB HSSD DB + CLIP + an OpenAI key, none of which are available
  in the audit environment. When unavailable, the **original asset is kept** and the
  request is recorded under `_vlmunr_worst_match` (per-object and on the state) so
  downstream tooling can see what was intended. Nothing crashes; the scene is preserved.

## HSM coordinate conventions (the high-risk parts)

HSM world is **Y-up, meters**; Blender/bpa is **Z-up**. The renderer converts so that
**Blender Z == HSM Y**.

- **Position** `yup_to_zup_position(x, y, z) -> (x, -z, y)` (preserves handedness; up-axis
  maps to Blender Z).
- **Rotation**: HSM rotation is a single scalar in **degrees CCW about HSM Y, with 0°
  facing −Z**. After the Y-up→Z-up swap, this is a rotation about **Blender Z** by the
  same signed angle → `yup_yaw_to_zup_euler(deg) -> (0, 0, deg)`.

### The STK fix + column-major detail (`stk_scene_state.json`)

`stk_scene_state.json` is **always** written. Each object's
`transform.data` is a flat **16-float COLUMN-MAJOR** (`reshape order='F'`) 4×4 matrix.
HSM writes it as `M = FIX @ T`, where:

- `FIX = [[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]` (X negated, Y/Z swapped). **FIX is
  its own inverse.**
- `T` holds the translation column `[-x, y, z]` (X pre-flipped) and a CCW-about-Y rotation
  in its 3×3 block.

Decode (`decode_stk_transform`):
1. `M = reshape(data, (4,4), order='F')`
2. `T = FIX @ M` (undo the fix)
3. HSM position = `(-T[0,3], T[1,3], T[2,3])`
4. rotation_deg = `atan2(T[2,0], T[0,0])`

This decode is verified against a known pose in the unit tests (encode → decode →
exact recovery of position and rotation).

`stk_scene_state.json` is **geometry-only** (no category/dims). Mesh resolution for the
stk path uses `modelId == "fpModel.<hssd_id>"` →
`<HSSD_DIR>/objects/<id[0]>/<id>.glb` (decomposed parts go under
`objects/decomposed/<base>/`), matching `hsm_core.retrieval.utils.mesh_paths`.

### `hsm_scene_state.json` (preferred, but conditional)

Richer file with `scene_objects` (each: `name`, `position` Y-up meters, `dimensions`,
`rotation` deg about Y, `mesh_path` absolute `.glb`, `id`, `obj_type`). The renderer
**prefers** this file (direct mesh_path + pose, no HSSD dir needed) and falls back to
`stk_scene_state.json` when absent. Note: on disk `scene_objects` is a **dict keyed by
id**; both dict and list forms are handled.

> **IMPORTANT — emitting `hsm_scene_state.json`:** HSM only writes this file when the
> scene is saved with `save_scene_state=True`. `main.py` does **not** expose this as a
> CLI flag, and the pipeline's final `scene.save(...)` calls
> (`hsm_core/scene/processing/scene_pipeline.py` lines ~141/145) use the default
> (`False`). To emit `hsm_scene_state.json`, patch those `.save()` calls — or the default
> of `hsm_core/scene/core/manager.py::save` — to pass `save_scene_state=True`. Without
> that patch only `stk_scene_state.json` is produced and you must set `HSSD_DIR` for the
> renderer to resolve meshes.

## VERIFICATION STATUS

**Verified (this environment):**
- Unit suite: **51 passed** (`pytest tests/ -q`). Covers exact filename strings;
  removal+renumber on synthetic stk **and** hsm states (counts == `round(n/k)`, ≥ 1,
  contiguous ids, deterministic across seeds, dict & list `scene_objects`); worst-match
  graceful degradation; and **PURE** Y-up→Z-up position/rotation conversion and the
  column-major 4×4 STK decode against **known inputs** (encode→decode exact recovery).
- **Synthetic bpy smoke render**: one config rendered on a primitive cube via
  `vlmunr_bpa` (bpy 5.1.2), producing a non-empty PNG. Run in a subprocess inside the
  test because bpa's fd-level `redirect_stdout` corrupts pytest's terminal writer
  in-process.
- `vlmunr_variants.py` CLI exercised end-to-end on a synthetic `hsm_scene_state.json`
  (all 6 variant files written).

**NOT validated (requires assets not present / not downloaded):**
- **Real-asset coordinate correctness on HSSD** — the Y-up→Z-up placement and STK decode
  are unit-tested with known matrices, but a full render of real HSSD `.glb` meshes (to
  confirm visual orientation/placement) needs the 72 GB HSSD dataset and was not run.
- **Worst-match retrieval** — needs the HSSD embeddings DB + CLIP + an OpenAI key. The
  hook is structured and degrades gracefully; the actual lowest-CLIP swap path is not
  exercised here.
- **Full HSM scene generation** (`gen.sh`) — not run (requires HSM's model/API deps).
