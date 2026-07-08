# VLM-Unreliability Integration Layer (HSM)

A self-contained rendering + content-variant layer added on top of HSM (Hierarchical
Scene Motifs) for the VLM-evaluator audit harness. HSM generates scene *geometry/layout*
but does **not** render RGB — this layer adds a headless Blender (bpy) renderer, a fixed
factor-sweep contract, and content-variant generators.

## Files added (all at repo root)

| File | Purpose |
|------|---------|
| `vlmunr_bpa.py` | Blender helper library (`Builder`, `Renderer`, `clear`, `initialize`, `import_obj`, `transform`). Copied from the `vlm-unreliability` repo with a GLB-import fix. |
| `vlmunr_hdri/` | 8 HDRI environment maps (`*.exr`) + `license.txt`. |
| `vlmunr_config.py` | Factor levels, white baseline, `RenderConfig`, `single_render_config()` + `render_all_configs()` (deduped six-factor sweep). |
| `vlmunr_render.py` | Scene parsing (PURE: STK decode, arch/shell spec, wall-quad tiling) + headless bpy renderer (`render_configs`): transparent master + per-bg composites. |
| `vlmunr_shell.py` | Dollhouse room-shell builder: floor (opaque) + walls with door/window cutout openings, backface-culling material (per-normal transparency). |
| `vlmunr_variants.py` | Removal + worst-match + within/cross substitution + layout-scramble content-variant generators (PURE + lazy retrieval hook). |
| `cli.py` | End-to-end driver: `--prompt` (generate + optional variants + optional render) or `--path` (render existing). `--render` / `--render-all`. |
| `tests/test_vlmunr_integration.py` | pytest: filenames, RenderConfig sweeps, STK encode/decode, shell tiling/arch parsing, bpy smoke renders. |
| `gen.sh` | Scene-generation driver (loops prompts -> `main.py`). |
| `run.sh` | Per-subdir render driver (loops scene dirs -> `vlmunr_render.py --mode`). |

## Run commands

```bash
PY=/Users/anson/miniforge3/envs/vlmunr/bin/python

# 1. Generate scenes (writes results/<name>/)
./gen.sh

# 2. Render every subfolder of a run dir (base/ + variant_*/)
./run.sh                    # mode 'all' (full six-factor sweep)
./run.sh single             # baseline only
RUN_DIR=outputs/20260708-023434 ./run.sh all

# Or invoke the renderer directly on one subfolder:
$PY vlmunr_render.py --scene-dir outputs/20260708-023434/base --mode all
$PY vlmunr_render.py --scene-dir results/bedroom_01            --mode single
# For stk-only scenes, set HSSD_DIR so meshes resolve:
HSSD_DIR=/path/to/hssd-models $PY vlmunr_render.py --scene-dir results/bedroom_01 --mode all

# Tests
$PY -m pytest tests/ -q
```

### Single-command generation + variants (`cli.py`)

`cli.py` generates a scene from a text prompt end-to-end and (optionally)
writes the four content variants derived from the base scene **without
regenerating or re-calling the LLM**, and (optionally) renders the scene(s) with
Blender. Run it under the `hsm` conda env (it needs HSM's full deps); the
`vlmunr` env only has bpy, so cli.py shells out to that interpreter for rendering
(override with `--vlmunr-python` / `VLMUNR_PYTHON`).

```bash
conda activate hsm
# generate + variants + render the baseline of each scene:
python cli.py --prompt "A cozy bedroom with a bed, nightstand, and a wardrobe." \
    --base-url https://api.openai.com/v1 --api-key sk-... \
    --model gpt-4o-2024-08-06 --temperature 0.7 --variants --render
# generate + variants + render the FULL six-factor sweep:
python cli.py --prompt "..." --variants --render-all
# render an EXISTING run (no generation) — base/ + every variant_*:
python cli.py --path outputs/20260708-023434 --render-all
```

Flags: `--prompt` (generate) **or** `--path` (render existing) — exactly one;
`--base-url`, `--api-key`, `--model`, `--temperature`; `--hssd-dir`,
`--data-dir` (dataset overrides); `--output` (default `outputs`); `--variants`
(generate the four content variants); `--seed` (default 42); `--render`
(baseline render) **or** `--render-all` (six-factor sweep) — mutually exclusive;
`--vlmunr-python` (bpy interpreter).
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

Written under `<scene-dir>/renderings/`. Each config renders one RGBA transparent
master (env-map-lit, transparent film), then alpha-composites every requested
background color over it via PIL (bpa's two-stage recipe). `fit_ratio=1` (tight-
fit) is always used.

- **Transparent master** (per resolution/focal/pitch/yaw/HDRI):
  `render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}.png`
  e.g. `render_res-512_focal-50_pitch-0_yaw-0_env-city.png`
- **Composited per background** (master alpha-composited over an RGB color):
  `render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}_bg-{r}-{g}-{b}.png`
  e.g. `render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png`

The filename pitch value is always the literal `rotation[0]` passed to
`render_perspective` (`pitch 0` == top-down), so names are consistent across
methods regardless of any per-method convention.

## Factor levels + sweeps (`vlmunr_config.py`)

Levels match paper Table 1.

| Factor | Levels | Baseline |
|--------|--------|----------|
| `RESOLUTIONS` | 196, 224, 256, 336, 384, 448, 512, 768, 1024 (9) | 512 |
| `FOCAL_LENGTHS` | 16, 24, 35, 50, 85, 100, 200 (7) | 50 |
| `BACKGROUNDS` | (0,0,0),(65,65,65),**(118,118,118)**,(128,128,128),(186,186,186),(204,204,204),(255,255,255),(255,0,0),(0,255,0),(0,0,255) (10) | (255,255,255) white |
| `HDRIS` | city, courtyard, forest, interior, night, studio, sunrise, sunset (8) | city |
| `PITCHES` | 0, 15, 30, 45, 60, 75, 90 (7) | 0 |
| `YAWS` | 0, 45, 90, 135, 180, 225, 270, 315 (8, 45° steps) | 0 |
| `BASELINE_YAW_PITCH` | 45 — pitch fixed at 45 when sweeping yaw alone (a top-down view is rotation-invariant about yaw) | — |

- `single_render_config()` → the one `--render` config: baseline (white bg), 512,
  50mm, pitch 0 (top-down), yaw 0, city.
- `render_all_configs()` → the six `--render-all` sweeps, **deduped** per camera
  config (res, focal, pitch, yaw, env): a config rendered by several sweeps is
  rendered once as a transparent master and composited over the UNION of every
  background color requested for it. The baseline config is touched by every
  sweep + the bg sweep, so it collects all 10 backgrounds.
  Sweeps: (1) resolution, (2) focal length, (3) pitch (yaw 0), (4) yaw (pitch
  45), (5) env map, (6) background.

A `RenderConfig` is one transparent master + its list of bg colors; its `key`
property is the dedup identity.

## Dollhouse room shell

A generated scene is more than furniture: it has a floor, walls, and often doors
and windows; object placements are defined relative to that shell. The renderer
keeps each method's architectural geometry (floor footprint, walls, door/window
openings) rather than rendering objects in a void. Opaque walls would occlude the
interior during oblique pitch/yaw sweeps, so walls use the **dollhouse
convention**: a backface-culling material makes the camera-facing (near) wall
faces transparent (driven by the surface normal via Blender's `Backfacing`
geometry node), while far walls — and the doors/windows on them — stay visible.
At top-down (pitch 0) walls are edge-on and negligible; as pitch increases the
near walls fall away automatically. Doors and windows are real cutout openings in
the wall quad mesh, so they are retained throughout (part of the wall geometry,
not discarded).

The PURE geometry lives in `vlmunr_render.py`:
- `parse_shell_spec(state, fmt)` → `{floor: [(x,y)...], walls: [{a,b,height,openings}]}`.
  stk uses `scene.arch.elements` (Floor polygon + Wall holes with precomputed
  boxes); hsm uses `room_vertices` + `door_location` + `window_location` with the
  HSM default door/window dimensions.
- `wall_quad_tiles(a, b, height, openings, centroid)` tiles a wall edge into quads
  that avoid every opening, wound outward (away from the floor centroid).
- `shell_wall_quads` / `shell_floor_verts3d` produce the final mesh verts.

`vlmunr_shell.build_shell(bpy, spec)` assembles the floor (opaque) + walls
(backface-culling material) into the Blender scene.

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
- Unit suite: **89 passed, 1 skipped** (`pytest tests/ -q`, vlmunr env). Covers
  the new filename strings (`render_res-.._focal-.._pitch-.._yaw-.._env-..[_bg-..]`);
  factor-level counts/values (RES 9, FOCAL 7, PITCH 7, YAW 8, BGS 10, HDRI 8);
  the white baseline `single_render_config`; the deduped `render_all_configs`
  (every sweep level present, baseline unions all 10 bgs, yaw sweep uses pitch
  45, non-baseline configs white-only); the dollhouse shell PURE geometry
  (`parse_shell_spec` for stk arch + hsm room_vertices/door/window,
  `wall_quad_tiles` solid-wall + cutout-openings, outward winding, CCW floor);
  removal+renumber on synthetic stk **and** hsm states (counts == `round(n/k)`,
  ≥ 1, contiguous ids, deterministic across seeds, dict & list `scene_objects`);
  worst-match graceful degradation; within/cross substitution intent recording
  + degradation on both formats; layout scramble on both formats; and the
  column-major 4×4 STK decode/encode round-trip against **known inputs**.
- **Synthetic bpy smoke renders** (bpy 5.1.2, subprocess): (a) one config on a
  primitive cube via `vlmunr_bpa`; (b) `render_configs` on a synthetic hsm scene
  building the dollhouse shell (floor + walls with a door & a window opening),
  writing the transparent master + white composite for a top-down (pitch 0) and
  an oblique (pitch 45) config — the oblique render shows ~35% transparent
  pixels (near walls culled) vs ~5% top-down (only the openings), confirming the
  backface-culling dollhouse behavior. Subprocess because bpa's fd-level
  `redirect_stdout` corrupts pytest's terminal writer in-process.
- `cli.py` arg validation: `--prompt`/`--path` mutual exclusion and
  `--render`/`--render-all` mutual exclusion; render-subfolder discovery
  (base/ + variant_* with state files, skipping empty variant dirs); bogus
  `--vlmunr-python` surfaces a clear error.
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
- **Full HSM scene generation** (`gen.sh` / `cli.py --prompt`) — not run (requires HSM's
  model/API deps + HSSD + CLIP + an OpenAI key).
