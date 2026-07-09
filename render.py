"""
render.py — Headless bpy renderer for HSM scenes (audit harness).

Renders an HSM scene under a set of factor configurations. For each render
config it first renders an RGBA *transparent master* (env-map-lit, transparent
film), then alpha-composites each requested background color over it via PIL —
exactly the bpa.py two-stage recipe.

Scene parsing (Y-up -> Z-up transform, column-major STK 4x4 decode, arch opening
extraction, wall-quad tiling) is factored into PURE functions at module top so
they can be unit-tested without bpy.

Filename contract (per config):
  transparent master : render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}.png
  composited (bg)   : render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}_bg-{r}-{g}-{b}.png

Dollhouse walls: the room shell (floor + walls-with-door/window-openings) is
built from the scene arch (see parse_shell_spec). Walls are flat double-sided
quads carrying a backface-culling material, so camera-facing (near) wall faces
render transparent (per surface normal) while far walls — with their door/window
openings — stay opaque. At the top-down default (pitch 0) walls are edge-on and
contribute negligibly. fit_ratio=1 (tight-fit) is always used.

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

import render_config as cfg

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

HDRI_DIR = Path(__file__).resolve().parent / "hdri"
HDRI_STRENGTH = 1.0

# Default architectural dimensions (HSM Y == Blender Z, vertical, meters).
# Used when building the shell from an hsm_scene_state.json that carries only
# room_vertices + door/window LOCATIONS (no precomputed hole boxes).
DEFAULT_WALL_HEIGHT = 2.5
DEFAULT_DOOR_WIDTH = 0.9
DEFAULT_DOOR_HEIGHT = 2.0
DEFAULT_WINDOW_WIDTH = 1.6
DEFAULT_WINDOW_HEIGHT = 1.2
DEFAULT_WINDOW_BOTTOM_HEIGHT = 1.0

# Always tight-fit per the audit spec (bpa fit_ratio=1).
FIT_RATIO = 1.0

# ===========================================================================
# PURE FUNCTIONS (no bpy) — unit-tested
# ===========================================================================


def yup_to_zup_position(
    position: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """HSM Y-up (x,y,z) -> Blender Z-up (x,-z,y) so Blender Z == HSM Y."""
    x, y, z = position
    return (x, -z, y)


def yup_yaw_to_zup_euler(rotation_deg: float) -> Tuple[float, float, float]:
    """HSM rotation (deg CCW about HSM Y) -> Blender Euler deg about Blender Z."""
    return (0.0, 0.0, float(rotation_deg))


def decode_stk_transform(
    data: List[float],
) -> Tuple[Tuple[float, float, float], float]:
    """Decode an on-disk stk object transform (flat 16-float COLUMN-MAJOR 4x4
    M = FIX @ T). Returns (hsm_position (x,y,z) Y-up meters, rotation_deg CCW about HSM Y)."""
    arr = np.asarray(data, dtype=float)
    if arr.size != 16:
        raise ValueError(f"STK transform must have 16 floats, got {arr.size}")
    M = arr.reshape((4, 4), order="F")
    T = STK_FIX_MATRIX @ M  # undo the fix (FIX is its own inverse)
    x = -float(T[0, 3])
    y = float(T[1, 3])
    z = float(T[2, 3])
    R = T[:3, :3]
    rot_deg = math.degrees(math.atan2(float(R[2, 0]), float(R[0, 0])))
    return (x, y, z), rot_deg


def encode_stk_transform(
    position: Tuple[float, float, float], rotation_deg: float
) -> List[float]:
    """Inverse of decode_stk_transform: encode HSM position + CCW-about-Y rotation
    into the on-disk flat 16-float COLUMN-MAJOR (order='F') stk transform."""
    x, y, z = position
    r = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(r), math.sin(r)
    T = np.eye(4)
    T[:3, :3] = np.array([[cos_r, 0.0, -sin_r], [0.0, 1.0, 0.0], [sin_r, 0.0, cos_r]])
    T[:3, 3] = [-float(x), float(y), float(z)]
    M = STK_FIX_MATRIX @ T
    return M.reshape(-1, order="F").tolist()


def stk_modelid_to_hssd_id(model_id: str) -> str:
    """Extract the HSSD id from a modelId like 'fpModel.<hssd_id>'."""
    return model_id.split(".")[-1]


def construct_hssd_mesh_path(hssd_dir: str, hssd_id: str) -> str:
    """Construct the HSSD .glb path: <hssd_dir>/objects/<id[0]>/<id>.glb."""
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


def parse_hsm_scene(state: dict) -> List[dict]:
    """PURE: parse hsm_scene_state.json into placement records (Blender Z-up)."""
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


def parse_stk_scene(state: dict, hssd_dir: Optional[str] = None) -> List[dict]:
    """PURE: parse stk_scene_state.json into placement records."""
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


def transparent_filename(
    res: int, focal: float, pitch: int, yaw: int, hdri: str
) -> str:
    """Transparent master render filename."""
    return f"render_res-{res}_focal-{focal}_pitch-{pitch}_yaw-{yaw}_env-{hdri}.png"


def composited_filename(
    res: int,
    focal: float,
    pitch: int,
    yaw: int,
    hdri: str,
    bg: Tuple[int, int, int],
) -> str:
    """Composited (per-background) render filename."""
    r, g, b = bg
    return (
        f"render_res-{res}_focal-{focal}_pitch-{pitch}_yaw-{yaw}_env-{hdri}"
        f"_bg-{r}-{g}-{b}.png"
    )


# ===========================================================================
# Shell spec (PURE): parse arch -> floor polygon + walls-with-openings
# ===========================================================================
#
# A ShellSpec dict: {"floor": [(x,y), ...], "walls": [ {a,b,height,openings}, ... ]}
#   - floor verts are in BLENDER XY (z=0).
#   - each wall: a,b = (x,y) Blender XY endpoints at floor; height = meters
#     (HSM Y == Blender Z); openings = [(u0,u1,z0,z1), ...] where u = distance
#     from `a` along the edge, z = height.
#
# Coordinate mapping:
#   stk arch point p = [hsm_x, hsm_z, 0]: Blender (x,y) = (p[0], -p[1]).
#   hsm room_vertex (x,z) -> Blender (x,-z) directly.

# Sentinel for an hsm-state scene with no architecture (objects in a void).
EMPTY_SHELL_SPEC = {"floor": [], "walls": []}


def _arch_point_to_blender_xy(p) -> Tuple[float, float]:
    """stk arch point [hsm_x, hsm_z, 0] -> Blender XY (x, -z)."""
    return (float(p[0]), -float(p[1]))


def _ensure_ccw_xy(
    verts: List[Tuple[float, float]],
) -> List[Tuple[float, float, float]]:
    """Return 3D floor verts (z=0) ordered CCW so the face normal points +Z."""
    pts = [(float(v[0]), float(v[1])) for v in verts]
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += (x1 - x0) * (y1 + y0)
    if s > 0:  # CW in this convention -> reverse
        pts = list(reversed(pts))
    return [(x, y, 0.0) for x, y in pts]


def _project_point_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float]:
    """Project (px,py) onto segment a->b. Returns (dist_from_a, closest_x, closest_y)."""
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    if L2 == 0.0:
        return 0.0, ax, ay
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L2))
    cx, cy = ax + t * abx, ay + t * aby
    return t * math.sqrt(L2), cx, cy


def _hsm_openings_for_walls(
    floor_verts_hsm: List[Tuple[float, float]],
    door_location,
    window_locations,
) -> Tuple[float, List[dict]]:
    """From hsm room_vertices + door/window LOCATIONS, build walls with default-sized
    openings. Returns (wall_height, walls) where walls use HSM XY endpoints + u/z
    (distances/heights are invariant under the yup_to_zup transform)."""
    n = len(floor_verts_hsm)
    walls = []
    for i in range(n):
        a = (float(floor_verts_hsm[i][0]), float(floor_verts_hsm[i][1]))
        b = (
            float(floor_verts_hsm[(i + 1) % n][0]),
            float(floor_verts_hsm[(i + 1) % n][1]),
        )
        walls.append({"a": a, "b": b, "height": DEFAULT_WALL_HEIGHT, "openings": []})

    # Door
    if door_location is not None:
        dx, dz = float(door_location[0]), float(door_location[1])
        best = None
        for w in walls:
            u, cx, cy = _project_point_on_segment(
                dx, dz, w["a"][0], w["a"][1], w["b"][0], w["b"][1]
            )
            off = math.hypot(dx - cx, dz - cy)
            if best is None or off < best[0]:
                best = (off, w, u)
        if best is not None and best[0] < 0.2:
            _, w, u = best
            w["openings"].append(
                (
                    u - DEFAULT_DOOR_WIDTH / 2,
                    u + DEFAULT_DOOR_WIDTH / 2,
                    0.0,
                    DEFAULT_DOOR_HEIGHT,
                )
            )

    # Windows
    if window_locations:
        for wloc in window_locations:
            wx, wz = float(wloc[0]), float(wloc[1])
            best = None
            for w in walls:
                u, cx, cy = _project_point_on_segment(
                    wx, wz, w["a"][0], w["a"][1], w["b"][0], w["b"][1]
                )
                off = math.hypot(wx - cx, wz - cy)
                if best is None or off < best[0]:
                    best = (off, w, u)
            if best is not None and best[0] < 0.2:
                _, w, u = best
                w["openings"].append(
                    (
                        u - DEFAULT_WINDOW_WIDTH / 2,
                        u + DEFAULT_WINDOW_WIDTH / 2,
                        DEFAULT_WINDOW_BOTTOM_HEIGHT,
                        DEFAULT_WINDOW_BOTTOM_HEIGHT + DEFAULT_WINDOW_HEIGHT,
                    )
                )
    return DEFAULT_WALL_HEIGHT, walls


def parse_shell_spec(state: dict, fmt: str) -> dict:
    """PURE: build a ShellSpec from a loaded scene state.

    stk: uses scene.arch.elements (Floor polygon + Wall holes with precomputed boxes).
    hsm: uses room_vertices + door_location + window_locations with default dims.
    """
    if fmt == "stk":
        arch = state.get("scene", {}).get("arch", {}) or {}
        elements = arch.get("elements", []) or []
        floor_el = next((e for e in elements if e.get("type") == "Floor"), None)
        wall_els = [e for e in elements if e.get("type") == "Wall"]
        floor = (
            [_arch_point_to_blender_xy(p) for p in floor_el.get("points", [])]
            if floor_el
            else []
        )
        walls = []
        for wel in wall_els:
            pts = wel.get("points", [])
            if len(pts) < 2:
                continue
            a = _arch_point_to_blender_xy(pts[0])
            b = _arch_point_to_blender_xy(pts[1])
            height = float(wel.get("height", DEFAULT_WALL_HEIGHT))
            openings = []
            for hole in wel.get("holes", []) or []:
                box = hole.get("box", {})
                mn = box.get("min", [0, 0])
                mx = box.get("max", [0, 0])
                openings.append(
                    (float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]))
                )
            walls.append({"a": a, "b": b, "height": height, "openings": openings})
        return {"floor": floor, "walls": walls}

    if fmt == "hsm":
        room_vertices = state.get("room_vertices")
        if not room_vertices or len(room_vertices) < 3:
            return dict(EMPTY_SHELL_SPEC)
        # Build walls in HSM (x,z) space; distances/heights are invariant.
        door_location = state.get("door_location")
        window_locations = state.get("window_location") or []
        _, walls_hsm = _hsm_openings_for_walls(
            [tuple(rv) for rv in room_vertices], door_location, window_locations
        )
        # Convert endpoints to Blender XY (x, -z).
        walls = []
        for w in walls_hsm:
            ax, az = w["a"]
            bx, bz = w["b"]
            walls.append(
                {
                    "a": (ax, -az),
                    "b": (bx, -bz),
                    "height": w["height"],
                    "openings": w["openings"],
                }
            )
        floor = [(float(rv[0]), -float(rv[1])) for rv in room_vertices]
        return {"floor": floor, "walls": walls}

    raise ValueError(f"Unknown scene-state format: {fmt!r}")


def wall_quad_normal_xy(a, b) -> Tuple[float, float]:
    """Outward normal (Blender XY, z=0) for CCW winding
    [P(u0,v0), P(u1,v0), P(u1,v1), P(u0,v1)] along edge a->b."""
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    return (dy, -dx)


def _wall_point(a, b, L, u, v) -> Tuple[float, float, float]:
    t = u / L if L else 0.0
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), v)


def wall_quad_tiles(
    a, b, height: float, openings, centroid
) -> List[List[Tuple[float, float, float]]]:
    """PURE: tile a wall (edge a->b in Blender XY, 0..height in Z) into quads
    that avoid every opening (u0,u1,z0,z1). Each quad is 4 (x,y,z) verts wound
    CCW so its normal points OUTWARD (away from `centroid`).

    Robust to arbitrary (incl. overlapping) openings via a u-segment x v-segment
    sweep: between consecutive u break-points, the wall is present in v-ranges
    not covered by any opening spanning that u-segment.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    L = math.hypot(bx - ax, by - ay)
    if L <= 1e-9 or height <= 1e-9:
        return []

    ups = sorted(
        {0.0, L} | {float(o[0]) for o in openings} | {float(o[1]) for o in openings}
    )
    nx, ny = wall_quad_normal_xy(a, b)
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    outward = (mx - centroid[0], my - centroid[1])
    flip = (nx * outward[0] + ny * outward[1]) < 0.0

    quads: List[List[Tuple[float, float, float]]] = []
    for k in range(len(ups) - 1):
        u_lo, u_hi = ups[k], ups[k + 1]
        if u_hi - u_lo <= 1e-9:
            continue
        u_mid = (u_lo + u_hi) / 2.0
        covering = [
            o
            for o in openings
            if float(o[0]) <= u_mid + 1e-9 and u_mid - 1e-9 <= float(o[1])
        ]
        vps = sorted(
            {0.0, height}
            | {float(o[2]) for o in covering}
            | {float(o[3]) for o in covering}
        )
        for j in range(len(vps) - 1):
            v_lo, v_hi = vps[j], vps[j + 1]
            if v_hi - v_lo <= 1e-9:
                continue
            v_mid = (v_lo + v_hi) / 2.0
            inside = any(
                float(o[2]) <= v_mid + 1e-9 and v_mid - 1e-9 <= float(o[3])
                for o in covering
            )
            if inside:
                continue
            p00 = _wall_point(a, b, L, u_lo, v_lo)
            p10 = _wall_point(a, b, L, u_hi, v_lo)
            p11 = _wall_point(a, b, L, u_hi, v_hi)
            p01 = _wall_point(a, b, L, u_lo, v_hi)
            quad = [p00, p10, p11, p01] if not flip else [p00, p01, p11, p10]
            quads.append(quad)
    return quads


def shell_floor_verts3d(spec: dict) -> List[Tuple[float, float, float]]:
    """Floor polygon as 3D verts ordered CCW (+Z normal), for mesh creation."""
    return _ensure_ccw_xy(spec.get("floor", []))


def shell_centroid(spec: dict) -> Tuple[float, float]:
    """Centroid of the floor polygon (for outward-normal determination)."""
    floor = spec.get("floor", [])
    if not floor:
        return (0.0, 0.0)
    n = len(floor)
    return (sum(v[0] for v in floor) / n, sum(v[1] for v in floor) / n)


def shell_wall_quads(spec: dict) -> List[List[Tuple[float, float, float]]]:
    """All wall quads (already outward-wound) across every wall in the spec."""
    cent = shell_centroid(spec)
    out: List[List[Tuple[float, float, float]]] = []
    for w in spec.get("walls", []):
        out.extend(
            wall_quad_tiles(w["a"], w["b"], w["height"], w.get("openings", []), cent)
        )
    return out


# ===========================================================================
# Scene loading
# ===========================================================================


def _detect_format(state: dict) -> str:
    if "scene_objects" in state:
        return "hsm"
    if "scene" in state and "object" in state.get("scene", {}):
        return "stk"
    raise ValueError("Unrecognized scene-state format")


def load_scene_records(scene_dir: Path, hssd_dir: Optional[str]) -> List[dict]:
    """Load placement records, preferring hsm_scene_state.json over stk.

    Returns an EMPTY list (not an error) when a state file exists but holds no
    objects — this is legitimate for shell-only renders (a scene whose objects
    failed to resolve, or a synthetic shell test). Raises only when NO state file
    is present at all."""
    hsm_path = scene_dir / "hsm_scene_state.json"
    stk_path = scene_dir / "stk_scene_state.json"
    if hsm_path.exists():
        with open(hsm_path) as f:
            state = json.load(f)
        return parse_hsm_scene(state)
    if stk_path.exists():
        with open(stk_path) as f:
            state = json.load(f)
        return parse_stk_scene(state, hssd_dir)
    raise FileNotFoundError(
        f"No hsm_scene_state.json or stk_scene_state.json under {scene_dir}"
    )


def load_shell_spec(scene_dir: Path) -> dict:
    """Load the room-shell spec. Prefers the stk arch (richer hole boxes) when
    present; falls back to hsm room_vertices + door/window locations."""
    stk_path = scene_dir / "stk_scene_state.json"
    hsm_path = scene_dir / "hsm_scene_state.json"
    if stk_path.exists():
        try:
            with open(stk_path) as f:
                state = json.load(f)
            if state.get("scene", {}).get("arch"):
                return parse_shell_spec(state, "stk")
        except Exception:
            pass
    if hsm_path.exists():
        with open(hsm_path) as f:
            state = json.load(f)
        return parse_shell_spec(state, "hsm")
    return dict(EMPTY_SHELL_SPEC)


# ===========================================================================
# bpy rendering (lazy import)
# ===========================================================================


def _hdri_path(hdri: str) -> Optional[Tuple[str, float]]:
    path = HDRI_DIR / f"{hdri}.exr"
    if path.exists():
        return (str(path), HDRI_STRENGTH)
    return None


def build_blender_scene(records: List[dict], shell_spec: dict) -> Tuple[object, object]:
    """Import all meshes into a fresh Blender scene, place them, and build the
    dollhouse room shell (floor + walls-with-openings, backface-culled).
    Returns (Renderer, (center, radius)) for camera framing."""
    import blender_bpa as bpa
    import render_shell as _vs

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
            print(f"[render] WARNING: import failed for {mp}: {_e}; skipped.")
            missing.append(mp)
            continue
        bpa.transform(obj, position=rec["position"], rotation=rec["rotation"])
        placed += 1
    if missing:
        print(
            f"[render] WARNING: {len(missing)} mesh(es) missing/unresolved; skipped."
        )

    # Dollhouse shell (floor + walls with door/window openings).
    try:
        _vs.build_shell(bpa.bpy, shell_spec)
    except Exception as _e:
        print(f"[render] WARNING: shell build failed: {_e}")

    renderer = bpa.Renderer()
    center, radius = renderer.compute_bounding_sphere()
    return renderer, (center, radius)


def render_configs(
    scene_dir: Path,
    configs: List[cfg.RenderConfig],
    hssd_dir: Optional[str] = None,
) -> List[str]:
    """Render every config for a scene. Builds the Blender scene once, then for
    each config: (re)init the world when the HDRI changes, cull walls per the
    camera pose is unnecessary (backface-culling material handles it), render the
    transparent master, then composite each requested background color.

    Returns the list of all written PNG paths (transparent masters + composites).
    """
    import blender_bpa as bpa

    records = load_scene_records(scene_dir, hssd_dir)
    shell_spec = load_shell_spec(scene_dir)
    out_dir = scene_dir / "renderings"
    out_dir.mkdir(exist_ok=True)

    outputs: List[str] = []
    if not configs:
        return outputs

    bpa.clear()
    renderer, (center, radius) = build_blender_scene(records, shell_spec)

    current_hdri: Optional[str] = None

    def ensure_world(hdri: str):
        nonlocal current_hdri
        if hdri != current_hdri:
            bpa.initialize(transparent=True, environment_map=_hdri_path(hdri))
            current_hdri = hdri

    for rc in configs:
        res, focal, pitch, yaw, hdri = (
            rc.resolution,
            rc.focal_length,
            rc.pitch,
            rc.yaw,
            rc.hdri,
        )
        ensure_world(hdri)
        master = out_dir / transparent_filename(res, focal, pitch, yaw, hdri)
        renderer.render_perspective(
            str(master),
            center,
            radius,
            rotation=(pitch, 0, yaw),
            resolution=res,
            focal_length=focal,
            fit_ratio=FIT_RATIO,  # tight-fit per audit spec
            background=None,  # transparent master
        )
        outputs.append(str(master))
        for bg in rc.backgrounds:
            comp = out_dir / composited_filename(res, focal, pitch, yaw, hdri, bg)
            bpa.Renderer.add_bg_to_rgba(str(master), str(comp), color=bg)
            outputs.append(str(comp))

    return outputs


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render HSM scenes for the VLM audit harness."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        type=Path,
        help="Directory containing hsm_scene_state.json and/or stk_scene_state.json",
    )
    parser.add_argument(
        "--mode",
        default=cfg.SINGLE_MODE,
        choices=cfg.ALL_MODES,
        help="single = one baseline render; all = the six factor sweeps (deduped)",
    )
    parser.add_argument(
        "--hssd-dir",
        default=os.environ.get("HSSD_DIR"),
        help="HSSD models root (for stk fallback mesh resolution)",
    )
    args = parser.parse_args()

    configs = (
        cfg.render_all_configs()
        if args.mode == cfg.ALL_MODE
        else [cfg.single_render_config()]
    )
    outs = render_configs(args.scene_dir, configs, args.hssd_dir)
    print(
        f"[render] {args.mode}: wrote {len(outs)} file(s) under {args.scene_dir}/renderings/"
    )


if __name__ == "__main__":
    main()
