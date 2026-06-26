"""
Integration tests for the VLM-unreliability rendering + variant layer.

Covers:
  (a) filename builders produce exact strings,
  (b) removal + renumber on synthetic stk AND hsm scene states,
  (c) PURE Y-up->Z-up transform and column-major 4x4 decode against known inputs,
  (d) a bpy SMOKE test rendering one config on a primitive cube via vlmunr_bpa.
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

import vlmunr_config as cfg
import vlmunr_render as rnd
import vlmunr_variants as var


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

def test_master_filename_exact():
    assert rnd.master_filename(512, 50, 0, 0, "city") == "render_512_50_0_0_city.png"
    assert rnd.master_filename(1024, 200, 60, 330, "sunset") == "render_1024_200_60_330_sunset.png"


def test_composite_filename_exact():
    assert rnd.composite_filename(512, 50, (128, 128, 128), 0, 0, "city") == \
        "render_512_50_128_128_128_0_0_city.png"
    assert rnd.composite_filename(224, 24, (0, 18, 65), 30, 90, "forest") == \
        "render_224_24_0_18_65_30_90_forest.png"


def test_phase_levels():
    assert cfg.phase_levels("1a")["vary"] == "resolution"
    assert cfg.phase_levels("1a")["levels"] == cfg.RESOLUTIONS
    assert cfg.phase_levels("1b")["levels"][0] == (0, 0, 0)
    assert cfg.phase_levels("1c")["levels"] == cfg.HDRIS
    assert cfg.phase_levels("1d")["levels"] == cfg.FOCAL_LENGTHS
    grid = cfg.phase_levels("2")["levels"]
    assert len(grid) == len(cfg.PITCHES) * len(cfg.YAWS)
    assert grid[0] == (0, 0)
    with pytest.raises(ValueError):
        cfg.phase_levels("9z")


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
    assert all("_vlmunr_worst_match" in o for o in objs)
    assert new_state["_vlmunr_worst_match"]["rank"] == 2


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
# (d) bpy SMOKE render
# ===========================================================================

SMOKE_SCRIPT = r'''
import sys, os
sys.path.insert(0, {root!r})
import vlmunr_bpa as bpa
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
    """Render one config on a primitive cube via vlmunr_bpa.

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
