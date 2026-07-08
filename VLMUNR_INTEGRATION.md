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
| `vlmunr_variants.py` | Removal + worst-match + within/cross substitution + layout-scramble content-variant generators (PURE + lazy retrieval hook). |
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
$PY vlmunr_render.py    --scene-dir results/bedroom_01 --phase 2_yaw
# For stk-only scenes, set HSSD_DIR so meshes resolve:
HSSD_DIR=/path/to/hssd-models $PY vlmunr_render.py --scene-dir results/bedroom_01 --phase 1a

# Tests
$PY -m pytest tests/ -q
```

### Single-command generation + variants (`cli.py`)

`cli.py` generates a scene from a text prompt end-to-end and (optionally)
writes the four content variants derived from the base scene **without
regenerating or re-calling the LLM**. Run it under the `hsm` conda env (it needs
HSM's full deps; the `vlmunr` env only has bpy).

```bash
conda activate hsm
python cli.py \
    --prompt "A cozy bedroom with a bed, nightstand, and a wardrobe." \
    --base-url https://api.openai.com/v1 --api-key sk-... \
    --model gpt-4o-2024-08-06 --temperature 0.7 --variants
```

Flags: `--prompt` (required), `--base-url`, `--api-key`, `--model`,
`--temperature`, `--output` (default `outputs`), `--variants`, `--seed` (default 42).
The LLM connection is wired through `VLMUNR_OPENAI_*` env vars (set by the CLI
from the flags), which `hsm_core.vlm.gpt.Session` reads at construction — so every
`create_session()` site across the pipeline picks them up without parameter
threading.

Output layout per run, under `outputs/<YYYYMMDD-HHMMSS>/` (UTC stamp):
`config.json` (prompt + all configs) at the top level; `base/` holding the
generated base scene (`room_scene.glb`, `hsm_scene_state.json` +
`stk_scene_state.json` — saved with `save_scene_state=True` so the richer
`hsm_scene_state.json` is emitted — plus `scene.log`, `visualizations/`,
`scene_motifs/`); and — when `--variants` — sibling subfolders
`variant_01_half/`, `variant_02_biggest-only/`, `variant_03_scrambled/`,
`variant_04_worst-object/`, each holding its own standalone
`hsm_scene_state.json` (canonical name) so it can be rendered independently
(`python vlmunr_render.py --scene-dir outputs/<stamp>/variant_03_scrambled`
writes `renderings/` into that subfolder).

The four variants: **01_half** keeps `round(n/2)` objects (seeded sample);
**02_biggest-only** keeps the single largest object by bbox volume;
**03_scrambled** randomizes every object's position AND heading in-room (rotation
too, per the spec); **04_worst-object** forks the saved state and re-runs HSM
retrieval with the CLIP ranking inverted (`worst_match=True`, see
`hsm_core/retrieval/core/retrieval_logic.py`) so each object's 3D asset is swapped
for the lowest-CLIP match — no LLM re-call. 04 degrades gracefully (keeps
originals, records intent under `_vlmunr_worst_match`) when HSSD/CLIP/OpenAI are
absent.

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

Levels match paper Table 1 exactly.

| Factor | Levels | Baseline |
|--------|--------|----------|
| `RESOLUTIONS` | 196, 224, 256, 336, 384, 448, 512, 768, 1024 (9) | 512 |
| `FOCAL_LENGTHS` | 16, 24, 35, 50, 85, 100, 200 (7) | 50 |
| `BACKGROUND_GRAYS` | 0, 65, 128, 186, 204, 255 (6) | (128,128,128) |
| `BACKGROUND_CHROMATIC` | (255,0,0), (0,255,0), (0,0,255) (3) | — |
| `FLOOR_TEXTURE_BACKGROUND` | `"floor_texture"` sentinel — documented as a Table 1 level but **NOT rendered** by this harness (textured-floor compositing is out of scope; recorded for completeness). | — |
| `HDRIS` | city, courtyard, forest, interior, night, studio, sunrise, sunset (8) | city |
| `PITCHES` | 0, 15, 30, 45, 60, 75, 90 (7) | 0 |
| `YAWS` | 0, 45, 90, 135, 180, 225, 270, 315 (8, 45° steps) | 0 |
| `BASELINE_YAW_PITCH` | 45 — pitch fixed at 45 when sweeping yaw alone | — |

`phase_levels(phase)` returns a **dict** with keys
`{resolution, focal_length, background, hdri, pitch, yaw, vary, levels}`:
- `1a` resolution sweep · `1b` background-gray sweep · `1b_chroma` chromatic-background
  sweep (R/G/B, 3 levels) · `1c` HDRI sweep · `1d` focal-length sweep
- `2` pitch × yaw camera-angle grid (7 × 8 = 56 poses, kept for backward compat)
- `2_pitch` pitch sweep (7 levels) with **yaw fixed at 0** (`BASELINE_YAW`)
- `2_yaw` yaw sweep (8 levels) with **pitch fixed at 45** (`BASELINE_YAW_PITCH`)

`ALL_PHASES = [1a, 1b, 1b_chroma, 1c, 1d, 2, 2_pitch, 2_yaw]` (drives `--phase all`).

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

**Substitution within/cross** (`variant_subst_within` / `variant_subst_cross`):
- Splits worst-match-style substitution into two content perturbations:
  `variant_subst_within` swaps each object's asset for a **different instance in the
  same category**; `variant_subst_cross` swaps it for an instance from a **different
  category**. Both go through the same lazy `hsm_core.retrieval` hook
  (`_try_substitution_lookup`).
- **Degrades gracefully**: when HSSD/CLIP are absent the original asset is kept and the
  intent is recorded as an `{object_id: mode}` map under `_vlmunr_substitution` (per-object
  and on the state, with `mode` ∈ {`within`, `cross`}). Handles both state formats.

**Layout scramble** (`variant_scramble`, pure, fully unit-tested):
- Relocates **every** object to a random position within the room footprint, preserving
  the object set, ids and rotation/orientation — destroying the arrangement while keeping
  the inventory.
- **stk format**: only the **translation** components of each column-major 4×4 transform
  are changed. It reuses `vlmunr_render.decode_stk_transform` / `encode_stk_transform`
  (same STK fix + Y-up + column-major `order='F'` convention as the renderer), so the 3×3
  rotation block (orientation) and the object's height (HSM Y) are preserved. Room bounds
  derive from the arch **Floor** element (falling back to **Wall** points, then to object
  position extents when no arch is present).
- **hsm format**: scrambles `position` x,z within the bounds derived from object position
  extents (hsm_scene_state has no arch), keeping `y` (height) and `rotation`. Dict/list
  `scene_objects` shape is preserved.
- Deterministic for a fixed `--seed`; a different seed yields a different layout.

`generate_all_variants` writes all nine variant files: 3 removal, 3 worst-match,
2 substitution, 1 scramble.

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
- Unit suite: **71 passed** (`pytest tests/ -q`). Covers exact filename strings;
  paper-Table-1 factor-level counts (RES 9, FOCAL 7, PITCH 7, YAW 8, GRAYS 6, CHROMA 3)
  and exact values; the new phases (`1b_chroma` → 3 chromatic levels, `2_pitch` → yaw==0
  with 7 levels, `2_yaw` → pitch==45 with 8 levels) plus preserved `phase_levels` dict
  shape; removal+renumber on synthetic stk **and** hsm states (counts == `round(n/k)`,
  ≥ 1, contiguous ids, deterministic across seeds, dict & list `scene_objects`);
  worst-match graceful degradation; within/cross substitution intent recording
  (`{object_id: mode}`) + degradation on both formats; layout scramble on both formats
  (determinism, preserved count/ids, in-bounds, height + rotation/orientation unchanged,
  arch-floor bounds with object-extent fallback); and **PURE** Y-up→Z-up position/rotation
  conversion and the column-major 4×4 STK decode/encode round-trip against **known
  inputs**.
- **Synthetic bpy smoke render**: one config rendered on a primitive cube via
  `vlmunr_bpa` (bpy 5.1.2), producing a non-empty PNG. Run in a subprocess inside the
  test because bpa's fd-level `redirect_stdout` corrupts pytest's terminal writer
  in-process.
- `vlmunr_variants.py` CLI exercised end-to-end on a synthetic `hsm_scene_state.json`
  (all 9 variant files written: 3 removal, 3 worst-match, 2 substitution, 1 scramble).

**NOT validated (requires assets not present / not downloaded):**
- **Real-asset coordinate correctness on HSSD** — the Y-up→Z-up placement and STK decode
  are unit-tested with known matrices, but a full render of real HSSD `.glb` meshes (to
  confirm visual orientation/placement) needs the 72 GB HSSD dataset and was not run.
- **Worst-match / substitution retrieval** — needs the HSSD embeddings DB + CLIP + an
  OpenAI key. The hooks are structured and degrade gracefully (worst-match records
  `_vlmunr_worst_match`; within/cross substitution records `_vlmunr_substitution` with an
  `{object_id: mode}` intent map); the actual asset-swap paths are not exercised here.
- **Full HSM scene generation** (`gen.sh`) — not run (requires HSM's model/API deps).
