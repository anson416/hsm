"""
vlmunr_render.py — Headless bpy renderer for HSM scenes (audit harness).

Builds an RGBA "master" render of an HSM scene under a fixed factor configuration,
then composites per-background-gray variants. Scene parsing (the Y-up -> Z-up
transform and the column-major STK 4x4 decode) is factored into PURE functions at
module top so they can be unit-tested without bpy.

Scene state inputs (under --scene-dir):
  * hsm_scene_state.json  (PREFERRED): scene_objects with mesh_path + position
    (Y-up meters) + rotation (deg CCW about Y, 0deg faces -Z) + dimensions.
  * stk_scene_state.json  (FALLBACK): column-major 4x4 transforms with the HSM
    STK fix applied (X negated, Y/Z swapped). modelId == "fpModel.<hssd_id>".

HSM world is Y-UP, meters; Blender/bpa is Z-UP. We convert so Blender Z == HSM Y.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import vlmunr_config as cfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The fix HSM applies when writing stk_scene_state.json (sv_fix_coordinates):
#   X negated, Y/Z swapped. The matrix is its own inverse.
STK_FIX_MATRIX = np.array(
    [
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)

HDRI_DIR = Path(__file__).resolve().parent / "vlmunr_hdri"
HDRI_STRENGTH = 1.0

# ===========================================================================
# PURE FUNCTIONS (no bpy) — unit-tested
# ===========================================================================


def yup_to_zup_position(position: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert an HSM Y-up position (x, y, z) to a Blender Z-up position.

    HSM: X=right, Y=up, Z=forward. Blender: X=right, Y=forward, Z=up.
    The standard Y-up -> Z-up mapping that preserves handedness is:
        x_b = x
        y_b = -z
        z_b = y
    so that Blender Z (up) == HSM Y (up).
    """
    x, y, z = position
    return (x, -z, y)


def yup_yaw_to_zup_euler(rotation_deg: float) -> Tuple[float, float, float]:
    """
    Map an HSM rotation (single scalar, degrees CCW about HSM Y) to a Blender
    Euler (degrees) about the Blender axes.

    Since Blender Z == HSM Y after the Y-up->Z-up conversion, a rotation about
    HSM Y becomes a rotation about Blender Z by the same signed angle.
    Returned as (rx, ry, rz) for bpa's transform(rotation=...).
    """
    return (0.0, 0.0, float(rotation_deg))


def decode_stk_transform(
    data: List[float],
) -> Tuple[Tuple[float, float, float], float]:
    """
    Decode an on-disk stk_scene_state.json object transform.

    The on-disk `data` is a flat 16-element COLUMN-MAJOR (reshape order='F') 4x4
    matrix M = FIX @ T, where T encodes position_flipped=[-x, y, z] in its
    translation column and a CCW-about-Y rotation in its 3x3 block.

    Returns:
        (hsm_position (x, y, z) Y-up meters, rotation_deg CCW about HSM Y)
    """
    arr = np.asarray(data, dtype=float)
    if arr.size != 16:
        raise ValueError(f"STK transform must have 16 floats, got {arr.size}")
    M = arr.reshape((4, 4), order="F")
    # Undo the STK fix (FIX is its own inverse).
    T = STK_FIX_MATRIX @ M
    # Translation column holds [-x, y, z].
    x = -float(T[0, 3])
    y = float(T[1, 3])
    z = float(T[2, 3])
    # Rotation about Y from the 3x3 block: rot = [[cos,0,-sin],[0,1,0],[sin,0,cos]].
    R = T[:3, :3]
    rot_deg = math.degrees(math.atan2(float(R[2, 0]), float(R[0, 0])))
    return (x, y, z), rot_deg


def encode_stk_transform(
    position: Tuple[float, float, float], rotation_deg: float
) -> List[float]:
    """
    Inverse of `decode_stk_transform`: encode an HSM Y-up position (x, y, z) and a
    CCW-about-Y rotation (degrees) into the on-disk flat 16-element COLUMN-MAJOR
    (reshape order='F') stk transform `data`.

    Builds T with translation column [-x, y, z] and the CCW-about-Y rotation block,
    then applies the STK fix (M = FIX @ T) and flattens column-major. Reusing the
    same STK_FIX_MATRIX guarantees decode(encode(p, r)) == (p, r).
    """
    x, y, z = position
    r = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(r), math.sin(r)
    T = np.eye(4)
    T[:3, :3] = np.array(
        [[cos_r, 0.0, -sin_r], [0.0, 1.0, 0.0], [sin_r, 0.0, cos_r]]
    )
    T[:3, 3] = [-float(x), float(y), float(z)]
    M = STK_FIX_MATRIX @ T
    return M.reshape(-1, order="F").tolist()


def stk_modelid_to_hssd_id(model_id: str) -> str:
    """Extract the HSSD id from a modelId like 'fpModel.<hssd_id>'."""
    return model_id.split(".")[-1]


def construct_hssd_mesh_path(hssd_dir: str, hssd_id: str) -> str:
    """
    Construct the HSSD .glb path: <hssd_dir>/objects/<id[0]>/<id>.glb
    (matches hsm_core.retrieval.utils.mesh_paths.construct_hssd_mesh_path).
    """
    if not hssd_id:
        raise ValueError("HSSD id cannot be empty")
    if "part" in hssd_id:
        base = hssd_id.split("_part_")[0]
        return str(Path(hssd_dir) / "objects" / "decomposed" / base / f"{hssd_id}.glb")
    return str(Path(hssd_dir) / "objects" / hssd_id[0] / f"{hssd_id}.glb")


def _iter_scene_objects(scene_objects):
    """hsm scene_objects may be a dict keyed by id or a list; yield the values."""
    if isinstance(scene_objects, dict):
        return list(scene_objects.values())
    return list(scene_objects)


def parse_hsm_scene(
    state: dict,
) -> List[dict]:
    """
    PURE: parse an hsm_scene_state.json dict into a list of placement records.

    Each record: {mesh_path, position (Blender Z-up), rotation (Blender euler deg),
                  dimensions, id, hssd_id}.
    """
    records = []
    for obj in _iter_scene_objects(state.get("scene_objects", [])):
        mesh_path = obj.get("mesh_path")
        pos = obj.get("position")
        if pos is None:
            continue
        rot = float(obj.get("rotation", 0.0) or 0.0)
        hssd_id = ""
        if mesh_path:
            hssd_id = os.path.splitext(os.path.basename(mesh_path))[0]
        records.append(
            {
                "mesh_path": mesh_path,
                "position": yup_to_zup_position(tuple(pos)),
                "rotation": yup_yaw_to_zup_euler(rot),
                "dimensions": obj.get("dimensions"),
                "id": obj.get("id"),
                "hssd_id": hssd_id,
            }
        )
    return records


def parse_stk_scene(
    state: dict, hssd_dir: Optional[str] = None
) -> List[dict]:
    """
    PURE: parse an stk_scene_state.json dict into a list of placement records.

    Decodes each column-major transform, undoes the STK fix, resolves mesh paths
    from modelId when an HSSD dir is provided.
    """
    records = []
    for obj in state.get("scene", {}).get("object", []):
        data = obj["transform"]["data"]
        (x, y, z), rot_deg = decode_stk_transform(data)
        hssd_id = stk_modelid_to_hssd_id(obj.get("modelId", ""))
        mesh_path = (
            construct_hssd_mesh_path(hssd_dir, hssd_id)
            if (hssd_dir and hssd_id)
            else None
        )
        records.append(
            {
                "mesh_path": mesh_path,
                "position": yup_to_zup_position((x, y, z)),
                "rotation": yup_yaw_to_zup_euler(rot_deg),
                "dimensions": None,
                "id": obj.get("id"),
                "hssd_id": hssd_id,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Filename builders (PURE) — unit-tested
# ---------------------------------------------------------------------------


def master_filename(
    res: int, focal: int, pitch: int, yaw: int, hdri: str
) -> str:
    """Transparent master render filename."""
    return f"render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png"


def composite_filename(
    res: int,
    focal: int,
    bg: Tuple[int, int, int],
    pitch: int,
    yaw: int,
    hdri: str,
) -> str:
    """Composited (per-background) render filename."""
    r, g, b = bg
    return f"render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png"


# ===========================================================================
# Scene loading
# ===========================================================================


def load_scene_records(scene_dir: Path, hssd_dir: Optional[str]) -> List[dict]:
    """
    Load placement records, preferring hsm_scene_state.json over stk_scene_state.json.
    """
    hsm_path = scene_dir / "hsm_scene_state.json"
    stk_path = scene_dir / "stk_scene_state.json"
    if hsm_path.exists():
        with open(hsm_path) as f:
            state = json.load(f)
        records = parse_hsm_scene(state)
        if records:
            return records
    if stk_path.exists():
        with open(stk_path) as f:
            state = json.load(f)
        return parse_stk_scene(state, hssd_dir)
    raise FileNotFoundError(
        f"No hsm_scene_state.json or stk_scene_state.json under {scene_dir}"
    )


# ===========================================================================
# bpy rendering (lazy import)
# ===========================================================================


def _hdri_path(hdri: str) -> Optional[Tuple[str, float]]:
    path = HDRI_DIR / f"{hdri}.exr"
    if path.exists():
        return (str(path), HDRI_STRENGTH)
    return None


_VLMUNR_WALLS = []
_VLMUNR_STATE = None


def _load_state_for_shell(scene_dir):
    """Load the stk/hsm scene-state JSON so the shell builder can read arch."""
    import json as _json, os as _os
    for name in ("stk_scene_state.json", "hsm_scene_state.json"):
        pth = _os.path.join(str(scene_dir), name)
        if _os.path.isfile(pth):
            try:
                return _json.load(open(pth))
            except Exception:
                return None
    return None


def _build_hsm_shell():
    """Build floor+walls from scene.arch. HSM arch point (a,b,c) -> Blender
    (a,-b,c); walls extrude +z to their height. Returns wall objects (tagged)."""
    try:
        import bpy  # noqa
        import vlmunr_shell as _vs
    except Exception:
        return []
    state = globals().get("_VLMUNR_STATE")
    if not state:
        return []
    arch = (state.get("scene", {}) or {}).get("arch", {}) or {}
    els = arch.get("elements", []) or []
    floor = next((e for e in els if e.get("type") == "Floor"), None)
    walls_src = [e for e in els if e.get("type") == "Wall"]
    if floor is None:
        return []
    # floor polygon: arch (a,b,c=0) -> Blender (a, -b)
    verts = [(float(p[0]), -float(p[1])) for p in floor.get("points", [])]
    if len(verts) < 3:
        return []
    wh = 2.5
    if walls_src:
        hs = [float(w.get("height", 2.5)) for w in walls_src if w.get("height")]
        if hs:
            wh = max(hs)
    try:
        return _vs.build_room_shell(bpy, verts, wh, margin=0.0, ceiling=False)
    except Exception as _e:
        print("VLMUNR hsm shell build failed:", _e)
        return []



def build_blender_scene(records: List[dict]) -> Tuple[object, object]:
    """
    Import all meshes into a fresh Blender scene and place them. Returns
    (Renderer, (center, radius)) for camera framing. bpy-dependent.
    """
    import vlmunr_bpa as bpa

    placed = 0
    missing = []
    for rec in records:
        mp = rec.get("mesh_path")
        if not mp or not os.path.exists(mp):
            missing.append(mp)
            continue
        try:
            obj = bpa.import_obj(mp)
        except Exception as _e:
            print(f"[vlmunr_render] WARNING: import failed for {mp}: {_e}; skipped.")
            missing.append(mp)
            continue
        bpa.transform(
            obj,
            position=rec["position"],
            rotation=rec["rotation"],
        )
        placed += 1
    if missing:
        print(f"[vlmunr_render] WARNING: {len(missing)} mesh(es) missing/unresolved; skipped.")
    # VLMUNR_PATCH room shell
    globals()["_VLMUNR_WALLS"] = _build_hsm_shell()
    renderer = bpa.Renderer()
    center, radius = renderer.compute_bounding_sphere()
    return renderer, (center, radius)


def render_phase(
    scene_dir: Path,
    phase: str,
    hssd_dir: Optional[str] = None,
) -> List[str]:
    """
    Render one phase sweep for a scene. Returns list of output PNG paths.
    Re-initializes the world whenever the HDRI changes.
    """
    import vlmunr_bpa as bpa

    records = load_scene_records(scene_dir, hssd_dir)
    globals()["_VLMUNR_STATE"] = _load_state_for_shell(scene_dir)
    out_dir = scene_dir / "renderings"
    out_dir.mkdir(exist_ok=True)

    spec = cfg.phase_levels(phase)
    outputs: List[str] = []

    bpa.clear()
    renderer, (center, radius) = build_blender_scene(records)

    current_hdri = None

    def ensure_world(hdri: str):
        nonlocal current_hdri
        if hdri != current_hdri:
            bpa.initialize(transparent=True, environment_map=_hdri_path(hdri))
            current_hdri = hdri

    vary = spec["vary"]
    for level in spec["levels"]:
        conf = dict(spec)
        if vary == "pitch_yaw":
            conf["pitch"], conf["yaw"] = level
        else:
            conf[vary] = level

        res = conf["resolution"]
        focal = conf["focal_length"]
        pitch = conf["pitch"]
        yaw = conf["yaw"]
        hdri = conf["hdri"]

        ensure_world(hdri)
        try:
            import vlmunr_shell as _vs
            _vs.cull_walls(globals().get("_VLMUNR_WALLS", []), pitch, yaw)
        except Exception:
            pass
        master = out_dir / master_filename(res, focal, pitch, yaw, hdri)
        renderer.render_perspective(
            str(master),
            center,
            radius,
            rotation=(pitch, 0, yaw),
            resolution=res,
            focal_length=focal,
            fit_ratio=0.6,  # VLMUNR: tighten framing toward per-vertex fit
            background=None,  # transparent master
        )
        outputs.append(str(master))

        # Composite per-background. For the background sweep, each level is the bg;
        # otherwise composite the single baseline background.
        if vary == "background":
            bgs = [conf["background"]]
        else:
            bgs = [conf["background"]]
        for bg in bgs:
            comp = out_dir / composite_filename(res, focal, bg, pitch, yaw, hdri)
            bpa.Renderer.add_bg_to_rgba(str(master), str(comp), color=bg)
            outputs.append(str(comp))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render HSM scenes for the VLM audit harness.")
    parser.add_argument("--scene-dir", required=True, type=Path,
                        help="Directory containing hsm_scene_state.json and/or stk_scene_state.json")
    parser.add_argument("--phase", default="all",
                        choices=cfg.ALL_PHASES + ["all"],
                        help="Audit phase to sweep")
    parser.add_argument("--hssd-dir", default=os.environ.get("HSSD_DIR"),
                        help="HSSD models root (for stk fallback mesh resolution)")
    args = parser.parse_args()

    phases = list(cfg.ALL_PHASES) if args.phase == "all" else [args.phase]
    for ph in phases:
        outs = render_phase(args.scene_dir, ph, args.hssd_dir)
        print(f"[vlmunr_render] phase {ph}: wrote {len(outs)} file(s)")


if __name__ == "__main__":
    main()
