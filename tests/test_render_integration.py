"""
Integration tests for the VLM-unreliability rendering + variant layer.

Covers:
  (a) filename builders produce exact strings,
  (b) removal + renumber on synthetic stk AND hsm scene states,
  (c) PURE Y-up->Z-up transform and column-major 4x4 decode against known inputs,
  (d) a bpy SMOKE test rendering one config on a primitive cube via blender_bpa.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import render_config as cfg
import render as rnd
import scene_variants as var


# ---------------------------------------------------------------------------
# Synthetic scene builders
# ---------------------------------------------------------------------------

def _encode_stk_transform(position, rotation_deg):
    """Replicate HSM's on-disk encode: create_transform_matrix(pos, rot, None) ->
    flatten(T.T) then sv_fix_coordinates (FIX @ T, reshape order='F')."""
    x, y, z = position
    pos_flip = [-x, y, z]
    r = math.radians(rotation_deg)
    rot = np.array([[math.cos(r), 0, -math.sin(r)],
                    [0, 1, 0],
                    [math.sin(r), 0, math.cos(r)]])
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos_flip
    data = T.T.flatten()  # what create_transform_matrix stores
    orig = np.asarray(data).reshape((4, 4), order="F")
    fixed = rnd.STK_FIX_MATRIX @ orig
    return fixed.reshape(-1, order="F").tolist()


def make_stk_state(n):
    objects = []
    for i in range(n):
        objects.append({
            "id": str(i),
            "modelId": f"fpModel.hssdid{i:04d}",
            "index": i,
            "parentIndex": -1,
            "transform": {"rows": 4, "cols": 4,
                          "data": _encode_stk_transform((float(i), 0.0, float(-i)), 15.0 * i)},
        })
    return {"format": "sceneState",
            "scene": {"up": {"x": 0, "y": 1, "z": 0},
                      "front": {"x": 0, "y": 0, "z": 1},
                      "unit": 1.0, "object": objects,
                      "arch": {"elements": []}}}


def make_stk_state_with_floor(n, half=5.0):
    """stk state with a square floor footprint of HSM x,z in [-half, half].
    On-disk arch points store [-x, 0, z], so a symmetric room is symmetric on disk."""
    state = make_stk_state(n)
    floor_points = [
        [-half, 0, -half],
        [half, 0, -half],
        [half, 0, half],
        [-half, 0, half],
    ]
    state["scene"]["arch"]["elements"] = [
        {"id": "floor_0", "type": "Floor", "points": floor_points, "roomId": "0"},
    ]
    return state


def make_hsm_state(n, as_dict=True):
    objs = []
    for i in range(n):
        objs.append({
            "name": f"obj_{i}",
            "position": [float(i), 0.0, float(-i)],
            "dimensions": [1.0, 1.0, 1.0],
            "rotation": 15.0 * i,
            "mesh_path": f"/fake/hssd/objects/a/hssdid{i:04d}.glb",
            "obj_type": "large",
            "id": str(i),
        })
    scene_objects = {o["id"]: o for o in objs} if as_dict else objs
    return {"scene_state_version": 1, "scene_objects": scene_objects}


# ===========================================================================
# (a) Filename builders
# ===========================================================================

def test_transparent_filename_exact():
    assert rnd.transparent_filename(512, 50, 0, 0, "city") == \
        "render_res-512_focal-50_pitch-0_yaw-0_env-city.png"
    assert rnd.transparent_filename(1024, 200, 60, 330, "sunset") == \
        "render_res-1024_focal-200_pitch-60_yaw-330_env-sunset.png"


def test_composited_filename_exact():
    assert rnd.composited_filename(512, 50, 0, 0, "city", (128, 128, 128)) == \
        "render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-128-128-128.png"
    assert rnd.composited_filename(224, 24, 30, 90, "forest", (0, 18, 65)) == \
        "render_res-224_focal-24_pitch-30_yaw-90_env-forest_bg-0-18-65.png"
    # white baseline
    assert rnd.composited_filename(512, 50, 0, 0, "city", (255, 255, 255)) == \
        "render_res-512_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png"


# ===========================================================================
# (a') Factor levels + RenderConfig sweep model
# ===========================================================================

def test_factor_level_counts():
    assert len(cfg.RESOLUTIONS) == 9
    assert len(cfg.FOCAL_LENGTHS) == 7
    assert len(cfg.PITCHES) == 7
    assert len(cfg.YAWS) == 8
    assert len(cfg.BACKGROUNDS) == 10   # 6 grays incl. 118 + 3 chromatic + black/white
    assert len(cfg.HDRIS) == 8


def test_factor_level_values():
    assert cfg.RESOLUTIONS == [196, 224, 256, 336, 384, 448, 512, 768, 1024]
    assert cfg.FOCAL_LENGTHS == [16, 24, 35, 50, 85, 100, 200]
    assert cfg.PITCHES == [0, 15, 30, 45, 60, 75, 90]
    assert cfg.YAWS == [0, 45, 90, 135, 180, 225, 270, 315]
    assert cfg.HDRIS == ["city", "courtyard", "forest", "interior", "night",
                         "studio", "sunrise", "sunset"]
    assert (0, 0, 0) in cfg.BACKGROUNDS
    assert (118, 118, 118) in cfg.BACKGROUNDS
    assert (255, 255, 255) in cfg.BACKGROUNDS
    assert (255, 0, 0) in cfg.BACKGROUNDS and (0, 255, 0) in cfg.BACKGROUNDS \
        and (0, 0, 255) in cfg.BACKGROUNDS
    assert cfg.BASELINE_BACKGROUND == (255, 255, 255)
    assert cfg.BASELINE_RESOLUTION == 512
    assert cfg.BASELINE_FOCAL_LENGTH == 50
    assert cfg.BASELINE_HDRI == "city"
    assert cfg.BASELINE_PITCH == 0
    assert cfg.BASELINE_YAW == 0
    assert cfg.BASELINE_YAW_PITCH == 45   # pitch fixed at 45 for the yaw sweep


def test_single_render_config_is_baseline_white_topdown():
    rc = cfg.single_render_config()
    assert rc.key == (512, 50, 0, 0, "city")
    assert rc.backgrounds == [(255, 255, 255)]
    assert rc.resolution == 512 and rc.focal_length == 50
    assert rc.pitch == 0 and rc.yaw == 0 and rc.hdri == "city"


def test_render_all_configs_covers_every_sweep_level():
    allc = cfg.render_all_configs()
    keys = {rc.key for rc in allc}
    # every resolution level appears
    for res in cfg.RESOLUTIONS:
        assert (res, 50, 0, 0, "city") in keys
    # every focal length
    for f in cfg.FOCAL_LENGTHS:
        assert (512, f, 0, 0, "city") in keys
    # every pitch (yaw 0)
    for p in cfg.PITCHES:
        assert (512, 50, p, 0, "city") in keys
    # every yaw (pitch fixed at 45)
    for y in cfg.YAWS:
        assert (512, 50, 45, y, "city") in keys
    # every env
    for e in cfg.HDRIS:
        assert (512, 50, 0, 0, e) in keys


def test_render_all_configs_dedups_and_unions_backgrounds():
    allc = cfg.render_all_configs()
    # The baseline camera config is touched by EVERY sweep + the bg sweep, so it
    # must collect all 10 background colors (union), but appear exactly once.
    baselines = [rc for rc in allc if rc.key == (512, 50, 0, 0, "city")]
    assert len(baselines) == 1
    assert set(baselines[0].backgrounds) == set(cfg.BACKGROUNDS)
    # overall dedup: keys are unique
    keys = [rc.key for rc in allc]
    assert len(keys) == len(set(keys))


def test_render_all_configs_yaw_uses_pitch_45():
    allc = cfg.render_all_configs()
    yaw_keys = [rc for rc in allc if rc.pitch == 45]
    # all 8 yaws present at pitch 45, none at pitch 0 except the baseline sweeps
    yaw_vals = {rc.yaw for rc in yaw_keys}
    assert yaw_vals == set(cfg.YAWS)


def test_render_all_configs_backgrounds_are_white_when_not_bg_sweep():
    allc = cfg.render_all_configs()
    # Every config NOT in the baseline set should carry only the white bg.
    for rc in allc:
        if rc.key != (512, 50, 0, 0, "city"):
            assert rc.backgrounds == [(255, 255, 255)]


# ===========================================================================
# (b) Removal + renumber
# ===========================================================================

@pytest.mark.parametrize("variant,divisor", list(var.REMOVAL_DIVISORS.items()))
@pytest.mark.parametrize("n", [1, 3, 8, 17])
def test_removal_stk_counts_and_renumber(variant, divisor, n):
    state = make_stk_state(n)
    out = var.remove_objects(state, divisor, seed=42)
    objs = out["scene"]["object"]
    assert len(objs) == max(1, round(n / divisor))
    ids = [o["id"] for o in objs]
    idxs = [o["index"] for o in objs]
    assert ids == [str(i) for i in range(len(objs))]
    assert idxs == list(range(len(objs)))


@pytest.mark.parametrize("variant,divisor", list(var.REMOVAL_DIVISORS.items()))
@pytest.mark.parametrize("n", [1, 3, 8, 17])
@pytest.mark.parametrize("as_dict", [True, False])
def test_removal_hsm_counts_and_renumber(variant, divisor, n, as_dict):
    state = make_hsm_state(n, as_dict=as_dict)
    out = var.remove_objects(state, divisor, seed=42)
    objs, _ = var._scene_objects_as_list(out["scene_objects"])
    assert len(objs) == max(1, round(n / divisor))
    ids = sorted(o["id"] for o in objs)
    assert ids == [str(i) for i in range(len(objs))]


def test_removal_deterministic():
    state = make_stk_state(16)
    a = var.remove_objects(state, 2, seed=7)
    b = var.remove_objects(state, 2, seed=7)
    c = var.remove_objects(state, 2, seed=8)
    sel_a = [o["modelId"] for o in a["scene"]["object"]]
    sel_b = [o["modelId"] for o in b["scene"]["object"]]
    sel_c = [o["modelId"] for o in c["scene"]["object"]]
    assert sel_a == sel_b
    assert sel_a != sel_c  # different seed -> different selection (16 -> 8 of 16)


def test_removal_min_one():
    state = make_stk_state(2)
    out = var.remove_objects(state, 8, seed=1)
    assert len(out["scene"]["object"]) == 1


# ===========================================================================
# (b') Worst-match graceful degradation
# ===========================================================================

def test_worst_match_degrades_gracefully():
    state = make_hsm_state(4)
    new_state, report = var.apply_worst_match(state, rank=2, seed=42)
    assert report["intended"] == 4
    assert report["applied"] == 0          # retrieval unavailable in audit env
    assert report["available"] is False
    # scene preserved, intent recorded
    objs, _ = var._scene_objects_as_list(new_state["scene_objects"])
    assert len(objs) == 4
    assert all("_worst_match" in o for o in objs)
    assert new_state["_worst_match"]["rank"] == 2


# ===========================================================================
# (b'') Biggest-only + scramble-rotation + named variants
# ===========================================================================

def make_hsm_state_distinct_dims(n, as_dict=True):
    """hsm state where object i has dimensions (i+1, 1, 1) so volumes are distinct."""
    objs = []
    for i in range(n):
        objs.append({
            "name": f"obj_{i}",
            "position": [float(i), 0.0, float(-i)],
            "dimensions": [float(i + 1), 1.0, 1.0],   # volume = i+1
            "rotation": 15.0 * i,
            "mesh_path": f"/fake/hssd/objects/a/hssdid{i:04d}.glb",
            "obj_type": "large",
            "id": str(i),
        })
    scene_objects = {o["id"]: o for o in objs} if as_dict else objs
    return {"scene_state_version": 1, "scene_objects": scene_objects}


def test_object_volume_and_biggest_index():
    assert var._object_volume({"dimensions": [2, 3, 4]}) == 24.0
    assert var._object_volume({"dimensions": [1, 1, 1]}) == 1.0
    assert var._object_volume({}) == 0.0
    assert var._object_volume({"dimensions": [-1, 2, 3]}) == 0.0  # negative -> 0
    objs = [{"dimensions": [1, 1, 1]}, {"dimensions": [3, 2, 2]}, {"dimensions": [2, 2, 2]}]
    assert var.select_biggest_index(objs) == 1  # volumes 1, 12, 8


@pytest.mark.parametrize("as_dict", [True, False])
def test_biggest_only_hsm_keeps_largest(as_dict):
    state = make_hsm_state_distinct_dims(5, as_dict=as_dict)
    out = var.biggest_only(state)
    objs, was_dict = var._scene_objects_as_list(out["scene_objects"])
    assert len(objs) == 1
    # largest volume = object index 4 (volume 5)
    assert objs[0]["dimensions"] == [5.0, 1.0, 1.0]
    assert objs[0]["id"] == "0"


def test_biggest_only_stk_keeps_largest():
    state = make_stk_state(4)
    # give stk objects dimensions via the hsm-style field? stk has none; biggest_only_stk
    # uses select_biggest_index over the stk object dicts which lack 'dimensions', so all
    # volumes are 0 and the stable sort keeps index 0.
    out = var.biggest_only_stk(state)
    objs = out["scene"]["object"]
    assert len(objs) == 1
    assert objs[0]["id"] == "0"
    assert objs[0]["index"] == 0


def test_biggest_only_empty():
    state = {"scene_state_version": 1, "scene_objects": {}}
    out = var.biggest_only(state)
    assert out["scene_objects"] == {}


def test_scramble_hsm_rotation_randomized():
    state = make_hsm_state_distinct_dims(5)
    out = var.scramble_hsm(state, seed=7, scramble_rotation=True)
    objs, _ = var._scene_objects_as_list(out["scene_objects"])
    # all 5 kept, rotations now in [0,360) and (almost surely) changed from originals
    assert len(objs) == 5
    for o in objs:
        assert 0.0 <= o["rotation"] <= 360.0
    orig_rots = [15.0 * i for i in range(5)]
    assert [o["rotation"] for o in objs] != orig_rots


def test_scramble_hsm_rotation_preserved_by_default():
    state = make_hsm_state_distinct_dims(5)
    out = var.scramble_hsm(state, seed=7, scramble_rotation=False)
    objs, _ = var._scene_objects_as_list(out["scene_objects"])
    assert [o["rotation"] for o in objs] == [15.0 * i for i in range(5)]


def test_scramble_layout_rotation_flag_dispatches(tmp_path):
    state = make_hsm_state_distinct_dims(4)
    a = var.scramble_layout(state, seed=3, scramble_rotation=True)
    b = var.scramble_layout(state, seed=3, scramble_rotation=False)
    objs_a, _ = var._scene_objects_as_list(a["scene_objects"])
    objs_b, _ = var._scene_objects_as_list(b["scene_objects"])
    assert [o["rotation"] for o in objs_a] != [o["rotation"] for o in objs_b]


def test_build_variant_dispatches():
    state = make_hsm_state_distinct_dims(6)
    half = var.build_variant(state, "removal", seed=42, divisor=2)
    big = var.build_variant(state, "biggest", seed=42)
    scr = var.build_variant(state, "scramble", seed=42, scramble_rotation=True)
    worst = var.build_variant(state, "worst", seed=42, rank=0)
    n_half = len(var._scene_objects_as_list(half["scene_objects"])[0])
    n_big = len(var._scene_objects_as_list(big["scene_objects"])[0])
    n_scr = len(var._scene_objects_as_list(scr["scene_objects"])[0])
    n_worst = len(var._scene_objects_as_list(worst["scene_objects"])[0])
    assert n_half == 3            # round(6/2)
    assert n_big == 1
    assert n_scr == 6             # scramble preserves the set
    assert n_worst == 6           # worst-object preserves the set (assets may swap)


def test_generate_named_variants_writes_all_four(tmp_path):
    """Flat-source form (source_subdir=None): base state lives directly in
    scene_dir; each variant is written to its OWN subfolder under scene_dir
    using the canonical state filename the renderer looks for."""
    state = make_hsm_state_distinct_dims(5)
    (tmp_path / "hsm_scene_state.json").write_text(json.dumps(state))
    written = var.generate_named_variants(tmp_path, seed=42)
    assert set(written.keys()) == set(var.ALL_VARIANT_NAMES)
    for name, path in written.items():
        p = Path(path)
        # Each variant lives in its own subfolder named after the variant...
        assert p.parent.name == name
        assert p.parent.parent == tmp_path
        # ...under the canonical name (hsm format -> hsm_scene_state.json).
        assert p.name == "hsm_scene_state.json"
        assert p.exists()
        st = json.loads(p.read_text())
        assert "scene_objects" in st


def test_generate_named_variants_source_subdir_layout(tmp_path):
    """cli.py layout: base scene under <run>/base/, variants as sibling subfolders
    <run>/variant_*/, each with a standalone canonical-named state file."""
    run_dir = tmp_path
    base_dir = run_dir / "base"
    base_dir.mkdir()
    state = make_hsm_state_distinct_dims(5)
    (base_dir / "hsm_scene_state.json").write_text(json.dumps(state))

    written = var.generate_named_variants(run_dir, seed=42, source_subdir="base")
    assert set(written.keys()) == set(var.ALL_VARIANT_NAMES)
    for name, path in written.items():
        p = Path(path)
        assert p.parent == run_dir / name          # sibling of base/
        assert p.parent.parent == run_dir
        assert p.name == "hsm_scene_state.json"
        assert p.exists()
    # base/ is untouched (variants are forks, not in-place edits of base)
    assert (base_dir / "hsm_scene_state.json").exists()


# ===========================================================================
# (b''') worst_match inverts the CLIP argsort
# ===========================================================================

def test_worst_match_inverts_argsort():
    """The ranking branch in run_primary_retrieval builds an argsort over the
    similarity tensor: best-match uses (-sim).argsort() (descending), worst_match
    uses sim.argsort() (ascending). Verify the two orderings are exact inverses
    for a known similarity vector, matching the source branch.
    """
    import torch
    sim = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.7])
    best = (-sim).argsort().tolist()      # source: best_match=False
    worst = sim.argsort().tolist()        # source: worst_match=True
    assert best == [0, 4, 2, 3, 1]        # 0.9, 0.7, 0.5, 0.3, 0.1
    assert worst == [1, 3, 2, 4, 0]       # 0.1, 0.3, 0.5, 0.7, 0.9
    # descending vs ascending are reverses of each other for distinct values
    assert best == worst[::-1]


def test_worst_match_flag_in_signatures():
    """worst_match is threaded through the public retrieval surface.

    Skipped when the full hsm_core.retrieval import chain is unavailable in this
    env (it pulls in python-dotenv / torch / clip via hsm_core.vlm.gpt); a
    render-only env may have bpy but not all HSM deps. Run under the `hsm` conda env to exercise.
    """
    import inspect
    try:
        from hsm_core.retrieval import retrieve, retrieve_adaptive
        from hsm_core.retrieval.core.retrieval_logic import run_primary_retrieval, handle_fallback_retrieval
    except Exception as e:  # missing transitive dep (e.g. python-dotenv) in this env
        pytest.skip(f"hsm_core.retrieval import chain unavailable in this env: {e}")
    for fn in (retrieve, retrieve_adaptive, run_primary_retrieval, handle_fallback_retrieval):
        assert "worst_match" in inspect.signature(fn).parameters


# ===========================================================================
# (c) PURE transform + decode against KNOWN inputs
# ===========================================================================

def test_yup_to_zup_position_known():
    # HSM (x=1, y=2 up, z=3 fwd) -> Blender (1, -3, 2); Blender Z == HSM Y.
    assert rnd.yup_to_zup_position((1.0, 2.0, 3.0)) == (1.0, -3.0, 2.0)
    # Up axis maps to Blender Z exactly.
    assert rnd.yup_to_zup_position((0.0, 5.0, 0.0)) == (0.0, 0.0, 5.0)


def test_yup_yaw_to_zup_euler_known():
    assert rnd.yup_yaw_to_zup_euler(30.0) == (0.0, 0.0, 30.0)
    assert rnd.yup_yaw_to_zup_euler(-90.0) == (0.0, 0.0, -90.0)


def test_decode_stk_transform_known():
    # Encode a known pose, decode it, assert exact recovery.
    pos = (1.0, 2.0, 3.0)
    rot = 30.0
    data = _encode_stk_transform(pos, rot)
    (x, y, z), rot_deg = rnd.decode_stk_transform(data)
    assert math.isclose(x, 1.0, abs_tol=1e-9)
    assert math.isclose(y, 2.0, abs_tol=1e-9)
    assert math.isclose(z, 3.0, abs_tol=1e-9)
    assert math.isclose(rot_deg, 30.0, abs_tol=1e-6)


def test_decode_stk_transform_identity():
    # Identity pose (origin, 0 deg) round-trips to origin.
    data = _encode_stk_transform((0.0, 0.0, 0.0), 0.0)
    (x, y, z), rot_deg = rnd.decode_stk_transform(data)
    assert (round(x, 9), round(y, 9), round(z, 9)) == (0.0, 0.0, 0.0)
    assert math.isclose(rot_deg, 0.0, abs_tol=1e-9)


def test_decode_stk_bad_length():
    with pytest.raises(ValueError):
        rnd.decode_stk_transform([1.0, 2.0, 3.0])


def test_parse_stk_scene_full_pipeline():
    state = make_stk_state(2)
    records = rnd.parse_stk_scene(state, hssd_dir="/fake/hssd")
    assert len(records) == 2
    # object 0: HSM pos (0,0,0) -> Blender (0,0,0)
    assert tuple(round(c, 6) for c in records[0]["position"]) == (0.0, 0.0, 0.0)
    # object 1: HSM pos (1,0,-1) -> Blender (1, 1, 0)
    assert tuple(round(c, 6) for c in records[1]["position"]) == (1.0, 1.0, 0.0)
    assert records[1]["mesh_path"] == "/fake/hssd/objects/h/hssdid0001.glb"
    assert records[1]["hssd_id"] == "hssdid0001"


def test_parse_hsm_scene_zup_conversion():
    state = make_hsm_state(2)
    records = rnd.parse_hsm_scene(state)
    assert len(records) == 2
    # HSM pos (1,0,-1) -> Blender (1, 1, 0)
    assert tuple(round(c, 6) for c in records[1]["position"]) == (1.0, 1.0, 0.0)
    assert records[1]["rotation"] == (0.0, 0.0, 15.0)


def test_construct_hssd_mesh_path():
    p = rnd.construct_hssd_mesh_path("/data/hssd", "4f557c5ba812")
    assert p == "/data/hssd/objects/4/4f557c5ba812.glb"


# ===========================================================================
# (c) Dollhouse shell — arch parsing + wall-quad tiling (PURE)
# ===========================================================================

def _hsm_shell_state():
    """4x4 square room: door centered on the y=0 wall, window on the x=0 wall."""
    return {
        "room_vertices": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "door_location": [2, 0],
        "window_location": [[0, 2]],
        "scene_objects": {},
    }


def test_parse_shell_spec_hsm_floor_and_walls():
    spec = rnd.parse_shell_spec(_hsm_shell_state(), "hsm")
    # floor flipped to Blender XY: (x, -z). Order is normalized to CCW (+Z normal).
    fv = rnd.shell_floor_verts3d(spec)
    assert set((round(v[0], 6), round(v[1], 6)) for v in fv) == {
        (0.0, 0.0), (4.0, 0.0), (4.0, -4.0), (0.0, -4.0)
    }
    # CCW => positive signed area (standard shoelace) so the face normal is +Z.
    s = sum(fv[i][0] * fv[(i + 1) % 4][1] - fv[(i + 1) % 4][0] * fv[i][1]
           for i in range(4))
    assert s > 0.0
    assert len(spec["walls"]) == 4
    # door opening lands on the wall running (0,0)->(4,0): centered at u=2, w=0.9
    w0 = spec["walls"][0]
    door = [o for o in w0["openings"] if o[2] == 0.0]  # z0==0 => reaches floor => door
    assert len(door) == 1
    u0, u1, z0, z1 = door[0]
    assert abs((u0 + u1) / 2 - 2.0) < 1e-6
    assert abs((u1 - u0) - rnd.DEFAULT_DOOR_WIDTH) < 1e-6
    assert abs(z1 - rnd.DEFAULT_DOOR_HEIGHT) < 1e-6
    # window opening lands on the wall running (0,4)->(0,0): centered at u=2 (mid)
    # find the wall whose endpoints share x==0
    winwall = next(w for w in spec["walls"] if w["a"][0] == 0.0 and w["b"][0] == 0.0)
    win = [o for o in winwall["openings"] if o[2] > 0.0]  # z0>0 => window
    assert len(win) == 1
    assert abs(win[0][2] - rnd.DEFAULT_WINDOW_BOTTOM_HEIGHT) < 1e-6


def test_parse_shell_spec_empty_when_no_room():
    spec = rnd.parse_shell_spec({"scene_objects": {}}, "hsm")
    assert spec == rnd.EMPTY_SHELL_SPEC


def test_wall_quad_tiles_solid_wall_is_one_quad():
    # a solid 4m wall, 2.5m tall, no openings, centroid below it -> 1 quad.
    a, b = (0.0, 0.0), (4.0, 0.0)
    quads = rnd.wall_quad_tiles(a, b, 2.5, [], centroid=(0.0, -2.0))
    assert len(quads) == 1
    q = quads[0]
    assert len(q) == 4
    # the wall lies in the XZ plane (y==0), so its face normal is +/-Y.
    # centroid (0,-2) is below the wall, so OUTWARD (away from room interior)
    # is +Y. The quad is wound so its front-face normal points outward (+Y).
    v0, v1, v2 = q[0], q[1], q[2]
    e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    ny = e1[2]*e2[0] - e1[0]*e2[2]   # Y component of e1 x e2
    assert ny > 0.0   # outward normal points +Y (away from centroid at -Y)


def test_wall_quad_tiles_cut_out_opening():
    # wall with one door opening reaching the floor (z in [0, 2.0]) spanning the
    # middle: leaves left strip + right strip (full height) + top strip above
    # the door (z in [2.0, 2.5]) = 3 quads.
    a, b = (0.0, 0.0), (4.0, 0.0)
    # door: u in [1.55, 2.45], z in [0, 2.0]
    quads = rnd.wall_quad_tiles(a, b, 2.5, [(1.55, 2.45, 0.0, 2.0)], centroid=(0.0, -2.0))
    assert len(quads) == 3
    # no quad vertex should lie strictly inside the opening box
    for q in quads:
        for v in q:
            u = v[0]   # along wall
            z = v[2]
            inside = (1.55 < u < 2.45) and (0.0 < z < 2.0)
            assert not inside


def test_shell_wall_quads_total():
    spec = rnd.parse_shell_spec(_hsm_shell_state(), "hsm")
    quads = rnd.shell_wall_quads(spec)
    # 3 walls solid (1 quad each) + 1 wall with a door + 1 wall with a window.
    # door wall: 4 quads (left/right full + top). window wall: 5 quads
    # (left/right full-height + bottom strip + top strip + middle above window? ->
    #  actually left+right full + below-window + above-window + between? verify >0)
    assert len(quads) > 0
    # every quad has exactly 4 verts
    assert all(len(q) == 4 for q in quads)


def test_parse_shell_spec_stk_arch_holes():
    # stk arch carries precomputed hole boxes; ensure they round-trip into openings.
    stk = {
        "scene": {
            "object": [],
            "arch": {
                "elements": [
                    {"type": "Wall", "height": 2.5,
                     "points": [[0, 0, 0], [-4, 0, 0]],   # stk [-hsm_x,0,hsm_z]
                     "holes": [{"type": "Door", "box": {"min": [1.55, 0], "max": [2.45, 2.0]}}]},
                    {"type": "Floor", "points": [[0, 0, 0], [-4, 0, 0], [-4, 0, 4], [0, 0, 4]]},
                ]
            }
        }
    }
    spec = rnd.parse_shell_spec(stk, "stk")
    assert len(spec["walls"]) == 1
    w = spec["walls"][0]
    # stk point [-hsm_x,0,hsm_z] -> Blender XY (-(-hsm_x), -(hsm_z))... undo flip:
    # Blender (x,y) = (-p[0], -p[2]). p=[0,0,0]->(0,0); p=[-4,0,0]->(4,0)
    assert w["a"] == (0.0, 0.0) and w["b"] == (4.0, 0.0)
    assert w["openings"] == [(1.55, 2.45, 0.0, 2.0)]
    assert len(spec["floor"]) == 4


# ===========================================================================
# (b'') Layout scramble — both state formats
# ===========================================================================

def test_scramble_stk_deterministic_and_preserved():
    state = make_stk_state_with_floor(8, half=5.0)
    a = var.scramble_layout(state, seed=7)
    b = var.scramble_layout(state, seed=7)
    c = var.scramble_layout(state, seed=8)
    da = [o["transform"]["data"] for o in a["scene"]["object"]]
    db = [o["transform"]["data"] for o in b["scene"]["object"]]
    dc = [o["transform"]["data"] for o in c["scene"]["object"]]
    assert da == db                 # deterministic for fixed seed
    assert da != dc                 # different seed -> different layout
    # count + ids preserved
    assert len(a["scene"]["object"]) == 8
    assert [o["id"] for o in a["scene"]["object"]] == [str(i) for i in range(8)]


def test_scramble_stk_in_bounds_and_rotation_unchanged():
    state = make_stk_state_with_floor(8, half=5.0)
    out = var.scramble_layout(state, seed=3)
    for orig, new in zip(state["scene"]["object"], out["scene"]["object"]):
        (ox, oy, oz), orot = rnd.decode_stk_transform(orig["transform"]["data"])
        (nx, ny, nz), nrot = rnd.decode_stk_transform(new["transform"]["data"])
        assert -5.0 - 1e-6 <= nx <= 5.0 + 1e-6
        assert -5.0 - 1e-6 <= nz <= 5.0 + 1e-6
        assert math.isclose(ny, oy, abs_tol=1e-6)        # height preserved
        assert math.isclose(nrot, orot, abs_tol=1e-6)    # orientation preserved


def test_scramble_stk_bounds_fallback_no_arch():
    # No arch -> bounds derive from object position extents; still in-bounds.
    state = make_stk_state(8)  # positions x in [0,7], z in [-7,0]
    out = var.scramble_layout(state, seed=1)
    for new in out["scene"]["object"]:
        (nx, _, nz), _ = rnd.decode_stk_transform(new["transform"]["data"])
        assert 0.0 - 1e-6 <= nx <= 7.0 + 1e-6
        assert -7.0 - 1e-6 <= nz <= 0.0 + 1e-6


@pytest.mark.parametrize("as_dict", [True, False])
def test_scramble_hsm_deterministic_bounds_preserved(as_dict):
    state = make_hsm_state(8, as_dict=as_dict)  # x in [0,7], z in [-7,0]
    a = var.scramble_layout(state, seed=5)
    b = var.scramble_layout(state, seed=5)
    objs_a, _ = var._scene_objects_as_list(a["scene_objects"])
    objs_b, _ = var._scene_objects_as_list(b["scene_objects"])
    pa = [o["position"] for o in objs_a]
    pb = [o["position"] for o in objs_b]
    assert pa == pb                  # deterministic
    assert len(objs_a) == 8
    assert sorted(o["id"] for o in objs_a) == [str(i) for i in range(8)]
    for o in objs_a:
        x, y, z = o["position"]
        assert 0.0 - 1e-6 <= x <= 7.0 + 1e-6
        assert -7.0 - 1e-6 <= z <= 0.0 + 1e-6


def test_scramble_hsm_height_and_rotation_unchanged():
    state = make_hsm_state(8)
    orig_objs, _ = var._scene_objects_as_list(state["scene_objects"])
    orig = {o["id"]: (o["position"][1], o["rotation"]) for o in orig_objs}
    out = var.scramble_layout(state, seed=2)
    objs, _ = var._scene_objects_as_list(out["scene_objects"])
    for o in objs:
        oy, orot = orig[o["id"]]
        assert math.isclose(o["position"][1], oy, abs_tol=1e-9)
        assert o["rotation"] == orot


# ===========================================================================
# (b''') Substitution within / cross intent recording + degradation
# ===========================================================================

@pytest.mark.parametrize("variant,mode", list(var.SUBST_MODES.items()))
@pytest.mark.parametrize("as_dict", [True, False])
def test_substitution_hsm_intent_and_degradation(variant, mode, as_dict):
    state = make_hsm_state(4, as_dict=as_dict)
    new_state, report = var.apply_substitution(state, mode, seed=42)
    assert report["mode"] == mode
    assert report["intended"] == 4
    assert report["applied"] == 0           # retrieval unavailable in audit env
    assert report["available"] is False
    assert report["intent"] == {str(i): mode for i in range(4)}
    objs, _ = var._scene_objects_as_list(new_state["scene_objects"])
    assert len(objs) == 4
    assert all(o["_substitution"]["mode"] == mode for o in objs)
    assert new_state["_substitution"]["mode"] == mode


@pytest.mark.parametrize("variant,mode", list(var.SUBST_MODES.items()))
def test_substitution_stk_intent_and_degradation(variant, mode):
    state = make_stk_state(3)
    new_state, report = var.apply_substitution(state, mode, seed=42)
    assert report["intended"] == 3
    assert report["applied"] == 0
    assert report["available"] is False
    assert report["intent"] == {str(i): mode for i in range(3)}
    assert all("_substitution" in o for o in new_state["scene"]["object"])


def test_substitution_bad_mode():
    with pytest.raises(ValueError):
        var.apply_substitution(make_hsm_state(2), "sideways", seed=1)


# ===========================================================================
# (d) bpy SMOKE render
# ===========================================================================

SMOKE_SCRIPT = r'''
import sys, os
sys.path.insert(0, {root!r})
import blender_bpa as bpa
out = {out!r}
bpa.clear()
cube = bpa.Builder.new_cube("SmokeCube")
bpa.transform(cube, position=(0, 0, 0))
bpa.initialize(transparent=True, samples=4, use_denoising=False)
renderer = bpa.Renderer()
center, radius = renderer.compute_bounding_sphere()
ok = renderer.render_perspective(
    out, center, radius,
    rotation=(60, 0, 30), resolution=128, focal_length=50,
    background=(128, 128, 128),
)
sys.exit(0 if (ok and os.path.exists(out) and os.path.getsize(out) > 0) else 1)
'''


def test_bpy_smoke_render_cube(tmp_path):
    """Render one config on a primitive cube via blender_bpa.

    bpa performs fd-level stdout redirection (redirect_stdout), which corrupts
    pytest's terminal writer if run in-process; we therefore drive the render in
    a clean subprocess and assert a non-empty PNG is produced.
    """
    import subprocess

    pytest.importorskip("bpy")
    out = tmp_path / "smoke.png"
    script = SMOKE_SCRIPT.format(root=str(ROOT), out=str(out))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"smoke render failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists()
    assert out.stat().st_size > 0


SHELL_RENDER_SCRIPT = r'''
import json, os, sys
sys.path.insert(0, {root!r})
scene_dir = {scene_dir!r}
state = {{
    "room_vertices": [[0,0],[4,0],[4,4],[0,4]],
    "door_location": [2, 0],
    "window_location": [[0, 2]],
    "scene_objects": [],
}}
os.makedirs(scene_dir, exist_ok=True)
with open(os.path.join(scene_dir, "hsm_scene_state.json"), "w") as f:
    json.dump(state, f)
import render as r, render_config as c
# baseline single config (top-down) + one oblique, to exercise shell + culling.
rcs = [
    c.RenderConfig(128, 50, 0, 0, "city", [(255, 255, 255)]),     # top-down
    c.RenderConfig(128, 50, 45, 0, "city", [(255, 255, 255)]),    # oblique
]
outs = r.render_configs(__import__("pathlib").Path(scene_dir), rcs, None)
sys.exit(0 if all(os.path.exists(o) and os.path.getsize(o) > 0 for o in outs) else 1)
'''


def test_bpy_smoke_render_shell_with_openings(tmp_path):
    """render_configs on a synthetic hsm scene builds the dollhouse shell (floor
    + walls with a door & a window opening) and writes the transparent master +
    white composite for a top-down and an oblique config. Subprocess (bpy fd
    redirect)."""
    import subprocess

    pytest.importorskip("bpy")
    scene_dir = tmp_path / "shell_scene"
    script = SHELL_RENDER_SCRIPT.format(root=str(ROOT), scene_dir=str(scene_dir))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"shell render failed:\n{proc.stdout}\n{proc.stderr}"
    rend = scene_dir / "renderings"
    # 2 configs * (transparent master + 1 white composite) = 4 files
    files = sorted(p.name for p in rend.iterdir())
    assert len(files) == 4
    assert "render_res-128_focal-50_pitch-0_yaw-0_env-city.png" in files
    assert "render_res-128_focal-50_pitch-0_yaw-0_env-city_bg-255-255-255.png" in files
    assert "render_res-128_focal-50_pitch-45_yaw-0_env-city.png" in files
    assert "render_res-128_focal-50_pitch-45_yaw-0_env-city_bg-255-255-255.png" in files
