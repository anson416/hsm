"""
scene_variants.py — Content-variant generators for the VLM audit harness.

Two families of variants, operating on whichever scene-state file exists:

  Removal (`variant_half` / `variant_quarter` / `variant_eighth`):
      keep round(n/k) objects (>=1), chosen by a seeded sample, then renumber
      ids/indices consistently. Pure, importable, fully unit-tested.

  Worst-match (`variant_alt_0` / `variant_alt_2` / `variant_alt_4`):
      swap each object's retrieved asset for a WORST-match (lowest-CLIP) asset
      via a lazy retrieval hook. Degrades gracefully: when models/assets are
      absent it records intent in the scene and keeps the original asset.

  Substitution (`variant_subst_within` / `variant_subst_cross`):
      swap each object's retrieved asset for a different instance in the SAME
      category (within) or an instance from a DIFFERENT category (cross), via the
      same lazy retrieval hook. Degrades gracefully: records intent
      {object_id: mode} and keeps the original asset when HSSD/CLIP are absent.

  Layout scramble (`variant_scramble`):
      relocate every object to a random position within the room footprint,
      preserving the object set, ids and rotation/orientation while destroying the
      arrangement. Pure, importable, deterministic for a fixed seed. Handles both
      the stk (column-major 4x4 transform) and hsm (position[x,y,z]) state formats.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import render as rnd

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

REMOVAL_DIVISORS = {"variant_half": 2, "variant_quarter": 4, "variant_eighth": 8}
WORST_MATCH_RANKS = {}  # alt_* dropped per methodology (keep within/cross only)
# Substitution modes: same category (different instance) vs different category.
SUBST_MODES = {"variant_subst_within": "within", "variant_subst_cross": "cross"}

# ---------------------------------------------------------------------------
# User-facing variant naming scheme (cli.py --variants)
# ---------------------------------------------------------------------------
# Maps the CLI's stable variant name -> (family, kwargs) used to build it. Each
# family resolves to one of the pure generator functions below.
VARIANT_SPECS = {
    "variant_01_half":          ("removal",   {"divisor": 2}),
    "variant_02_biggest-only":  ("biggest",   {}),
    "variant_03_scrambled":    ("scramble", {"scramble_rotation": True}),
    "variant_04_worst-object":  ("worst",     {"rank": 0}),
}
# Ordered list for deterministic generation order.
ALL_VARIANT_NAMES = list(VARIANT_SPECS.keys())


def detect_format(state: dict) -> str:
    """Return 'hsm' or 'stk' based on the loaded state dict."""
    if "scene_objects" in state:
        return "hsm"
    if "scene" in state and "object" in state.get("scene", {}):
        return "stk"
    raise ValueError("Unrecognized scene-state format")


def _scene_objects_as_list(scene_objects) -> Tuple[List[dict], bool]:
    """Normalize hsm scene_objects (dict-by-id or list) to a list; flag if dict."""
    if isinstance(scene_objects, dict):
        return list(scene_objects.values()), True
    return list(scene_objects), False


# ---------------------------------------------------------------------------
# Removal (PURE)
# ---------------------------------------------------------------------------


def kept_count(n: int, divisor: int) -> int:
    """Number of objects to keep = max(1, round(n / divisor))."""
    if n <= 0:
        return 0
    return max(1, round(n / divisor))


def select_keep_indices(n: int, divisor: int, seed: int) -> List[int]:
    """Deterministically choose which object indices to KEEP, sorted ascending."""
    k = kept_count(n, divisor)
    rng = random.Random(seed)
    chosen = rng.sample(range(n), k)
    return sorted(chosen)


def remove_objects_stk(state: dict, divisor: int, seed: int) -> dict:
    """
    PURE: drop objects from an stk-format state to round(n/k) (>=1) and renumber
    id/index contiguously (0..k-1). parentIndex left at -1 (HSM writes -1).
    """
    state = copy.deepcopy(state)
    objects = state["scene"]["object"]
    n = len(objects)
    keep = select_keep_indices(n, divisor, seed)
    new_objects = []
    for new_idx, old_idx in enumerate(keep):
        obj = objects[old_idx]
        obj["id"] = str(new_idx)
        obj["index"] = new_idx
        obj["parentIndex"] = -1
        new_objects.append(obj)
    state["scene"]["object"] = new_objects
    return state


def remove_objects_hsm(state: dict, divisor: int, seed: int) -> dict:
    """
    PURE: drop objects from an hsm-format state to round(n/k) (>=1) and renumber
    the surviving scene_objects' ids contiguously ('0'..'k-1'). Preserves whether
    scene_objects was a dict (re-keyed by new id) or a list.
    """
    state = copy.deepcopy(state)
    objs, was_dict = _scene_objects_as_list(state["scene_objects"])
    n = len(objs)
    keep = select_keep_indices(n, divisor, seed)
    new_list = []
    for new_idx, old_idx in enumerate(keep):
        obj = objs[old_idx]
        obj["id"] = str(new_idx)
        new_list.append(obj)
    if was_dict:
        state["scene_objects"] = {o["id"]: o for o in new_list}
    else:
        state["scene_objects"] = new_list
    return state


def remove_objects(state: dict, divisor: int, seed: int) -> dict:
    """Dispatch removal on detected format."""
    fmt = detect_format(state)
    if fmt == "stk":
        return remove_objects_stk(state, divisor, seed)
    return remove_objects_hsm(state, divisor, seed)


# ---------------------------------------------------------------------------
# Biggest-only (PURE)
# ---------------------------------------------------------------------------


def _object_volume(obj: dict) -> float:
    """Bounding-box volume of an object from its `dimensions` (width,height,depth).

    Falls back to 0.0 when dimensions are absent/invalid so the ordering is stable
    (the largest known object wins; ties broken by original order via stable sort).
    """
    dims = obj.get("dimensions")
    if not dims or not isinstance(dims, (list, tuple)) or len(dims) < 3:
        return 0.0
    try:
        w, h, d = float(dims[0]), float(dims[1]), float(dims[2])
    except (TypeError, ValueError):
        return 0.0
    if any(v < 0 for v in (w, h, d)):
        return 0.0
    return w * h * d


def select_biggest_index(objects: List[dict]) -> int:
    """Index of the largest object by bounding-box volume (stable, -1 if empty)."""
    if not objects:
        return -1
    best_idx, best_vol = 0, _object_volume(objects[0])
    for i in range(1, len(objects)):
        vol = _object_volume(objects[i])
        if vol > best_vol:
            best_vol, best_idx = vol, i
    return best_idx


def biggest_only_stk(state: dict) -> dict:
    """PURE: keep only the single largest object (by bbox volume), renumbered to id 0."""
    state = copy.deepcopy(state)
    objects = state["scene"]["object"]
    idx = select_biggest_index(objects)
    if idx < 0:
        state["scene"]["object"] = []
        return state
    obj = objects[idx]
    obj["id"] = "0"
    obj["index"] = 0
    obj["parentIndex"] = -1
    state["scene"]["object"] = [obj]
    return state


def biggest_only_hsm(state: dict) -> dict:
    """PURE: keep only the single largest hsm scene_object, renumbered to id '0'.

    Preserves whether scene_objects was a dict (re-keyed) or a list.
    """
    state = copy.deepcopy(state)
    objs, was_dict = _scene_objects_as_list(state["scene_objects"])
    idx = select_biggest_index(objs)
    if idx < 0:
        kept = []
    else:
        obj = objs[idx]
        obj["id"] = "0"
        kept = [obj]
    if was_dict:
        state["scene_objects"] = {o["id"]: o for o in kept}
    else:
        state["scene_objects"] = kept
    return state


def biggest_only(state: dict) -> dict:
    """Dispatch biggest-only on detected format."""
    fmt = detect_format(state)
    if fmt == "stk":
        return biggest_only_stk(state)
    return biggest_only_hsm(state)



# ---------------------------------------------------------------------------
# Layout scramble (PURE)
# ---------------------------------------------------------------------------


def _floor_bounds_stk(state: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Derive (min_x, max_x, min_z, max_z) room footprint in HSM Y-up coords from the
    stk arch floor/walls. stk arch points are on-disk [x, 0, z] (X pre-flipped); the
    HSM x is the negation of the stored x (see decode_stk_transform). We collect the
    floor element's points (falling back to walls), mapping each [sx, _, sz] to HSM
    (x=-sx, z=sz). Returns None if no arch geometry is present.
    """
    arch = state.get("scene", {}).get("arch") or {}
    elements = arch.get("elements") or []
    xs: List[float] = []
    zs: List[float] = []
    floor_pts: List[list] = []
    wall_pts: List[list] = []
    for el in elements:
        pts = el.get("points") or []
        if el.get("type") == "Floor":
            floor_pts.extend(pts)
        elif el.get("type") == "Wall":
            wall_pts.extend(pts)
    pts = floor_pts or wall_pts
    for p in pts:
        if len(p) >= 3:
            # stk arch floor points are stored as [px, py, 0] in the floor plane:
            # px maps to HSM x and py maps to HSM z (depth). This matches decoded
            # object positions (x in [0,W], z in [0,D]). Do NOT negate x and do
            # NOT read p[2] (always 0) — that put scrambled objects outside the
            # room and collapsed the z range to a single line.
            xs.append(float(p[0]))
            zs.append(float(p[1]))
    if not xs or not zs:
        return None
    return (min(xs), max(xs), min(zs), max(zs))


def _bounds_from_positions(positions: List[Tuple[float, float, float]]):
    """Fallback (min_x, max_x, min_z, max_z) from object position extents (Y-up)."""
    if not positions:
        return None
    xs = [p[0] for p in positions]
    zs = [p[2] for p in positions]
    return (min(xs), max(xs), min(zs), max(zs))


def _rand_in_bounds(rng: random.Random, bounds: Tuple[float, float, float, float]):
    """Sample a random (x, z) inside the [min_x,max_x] x [min_z,max_z] footprint."""
    min_x, max_x, min_z, max_z = bounds
    x = rng.uniform(min_x, max_x) if max_x > min_x else min_x
    z = rng.uniform(min_z, max_z) if max_z > min_z else min_z
    return x, z


def scramble_stk(state: dict, seed: int, scramble_rotation: bool = False) -> dict:
    """
    PURE: relocate every stk object to a random (x, z) within the room footprint,
    preserving the object set and ids/indices. By default only the translation
    components change (orientation kept via re-encoding the decoded rotation); pass
    ``scramble_rotation=True`` to also randomize each object's heading (deg CCW
    about Y) in [0, 360). Reuses decode/encode helpers so the column-major + Y-up
    convention matches the renderer. Deterministic for a fixed seed.
    """
    state = copy.deepcopy(state)
    objects = state["scene"]["object"]

    decoded = [rnd.decode_stk_transform(o["transform"]["data"]) for o in objects]
    positions = [pos for pos, _ in decoded]
    bounds = _floor_bounds_stk(state) or _bounds_from_positions(positions)
    if bounds is None:
        return state  # nothing to scramble (no geometry)

    rng = random.Random(seed)
    for obj, (pos, rot_deg) in zip(objects, decoded):
        new_x, new_z = _rand_in_bounds(rng, bounds)
        new_pos = (new_x, pos[1], new_z)  # keep height (HSM Y)
        new_rot = rng.uniform(0.0, 360.0) if scramble_rotation else rot_deg
        obj["transform"]["data"] = rnd.encode_stk_transform(new_pos, new_rot)
    return state


def scramble_hsm(state: dict, seed: int, scramble_rotation: bool = False) -> dict:
    """
    PURE: relocate every hsm scene_object to a random (x, z) within the room
    footprint, keeping y (height). By default rotation is preserved; pass
    ``scramble_rotation=True`` to also randomize each object's `rotation`
    (deg CCW about Y) in [0, 360). Bounds derive from object position extents
    (hsm_scene_state has no arch). Preserves dict/list shape and ids.
    Deterministic for a fixed seed.
    """
    state = copy.deepcopy(state)
    objs, was_dict = _scene_objects_as_list(state["scene_objects"])
    positions = [tuple(o["position"]) for o in objs if o.get("position") is not None]
    bounds = _bounds_from_positions(positions)
    if bounds is None:
        return state

    rng = random.Random(seed)
    for obj in objs:
        pos = obj.get("position")
        if pos is None:
            continue
        new_x, new_z = _rand_in_bounds(rng, bounds)
        obj["position"] = [new_x, float(pos[1]), new_z]  # keep y = height
        if scramble_rotation:
            obj["rotation"] = rng.uniform(0.0, 360.0)
    if was_dict:
        state["scene_objects"] = {o["id"]: o for o in objs}
    else:
        state["scene_objects"] = objs
    return state


def scramble_layout(state: dict, seed: int, scramble_rotation: bool = False) -> dict:
    """Dispatch layout scramble on detected format."""
    fmt = detect_format(state)
    if fmt == "stk":
        return scramble_stk(state, seed, scramble_rotation=scramble_rotation)
    return scramble_hsm(state, seed, scramble_rotation=scramble_rotation)


# ---------------------------------------------------------------------------
# Worst-match retrieval hook (lazy, graceful)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worst-object: real retrieval-hack fork (no LLM re-call, graceful)
# ---------------------------------------------------------------------------
#
# This forks the saved scene state and re-runs HSM's object retrieval with the
# CLIP ranking INVERTED (worst_match=True, see hsm_core.retrieval), so each
# object's 3D asset is swapped for the LOWEST-CLIP (worst) match. It does NOT
# re-run the LLM/scene-generation pipeline — only asset retrieval is replayed on
# lightweight Obj proxies built from the saved state. When the HSSD DB / CLIP /
# OpenAI key are absent (the audit environment), retrieval assigns no new meshes
# and we degrade gracefully: the original assets are kept and the intent is
# recorded under `_worst_match` so downstream tooling can see the request.


def _dims_to_half_size(dims) -> Optional[List[float]]:
    """SceneObject dimensions are (width, height, depth); Obj expects half-extents."""
    if not dims or not isinstance(dims, (list, tuple)) or len(dims) < 3:
        return None
    try:
        w, h, d = float(dims[0]), float(dims[1]), float(dims[2])
    except (TypeError, ValueError):
        return None
    return [w / 2.0, h / 2.0, d / 2.0]


def _build_obj_proxy(obj_state: dict, obj_type_enum) -> "Obj":
    """Build a lightweight Obj proxy from a saved scene-state object dict."""
    import numpy as np
    from hsm_core.scene_motif.core.bounding_box import BoundingBox
    from hsm_core.scene_motif.core.obj import Obj

    label = obj_state.get("name") or obj_state.get("obj_type") or "object"
    half = _dims_to_half_size(obj_state.get("dimensions")) or [0.1, 0.1, 0.1]
    bb = BoundingBox(centroid=[0.0, 0.0, 0.0], half_size=half, coord_axes=np.eye(3))
    return Obj(
        label=label,
        bounding_box=bb,
        id=obj_state.get("id"),
        description=obj_state.get("name") or "",
    )


def _try_worst_match_retrieval(
    obj_states: List[dict], rank: int, seed: int
) -> Tuple[Optional[List[Optional[str]]], dict]:
    """
    Replay retrieval with worst_match=True on Obj proxies built from `obj_states`.

    Returns (new_mesh_paths, info) where new_mesh_paths[i] is the worst-CLIP mesh
    path assigned to object i, or None if retrieval was unavailable / assigned
    nothing. `info` carries diagnostics (model_type, error). Heavy imports
    (torch, clip, hsm_core.retrieval) are lazy and isolated so a missing dep /
    missing HSSD DB / missing OpenAI key degrades to (None, {...}) instead of
    crashing the whole variant pass.
    """
    import asyncio
    import numpy as np  # noqa: F401  (some retrieval paths import numpy eagerly)

    info: dict = {"available": False, "error": None, "n_objs": len(obj_states)}
    if not obj_states:
        return [], info

    try:
        from hsm_core.scene.core.objecttype import ObjectType
        from hsm_core.retrieval import retrieve

        # All objects in a single-room scene share one retrieval pass; same_per_label
        # is False so distinct labels keep distinct worst-matches.
        proxies = [_build_obj_proxy(o, ObjectType.UNDEFINED) for o in obj_states]

        async def _run():
            await retrieve(
                objs=proxies,
                same_per_label=False,
                avoid_used=False,
                randomize=False,
                use_top_k=1,            # consider only the single worst candidate
                force_k=rank,           # if rank>0, force that worst-rank candidate
                worst_match=True,       # <-- the hack: invert (-sim).argsort()
                object_type=ObjectType.UNDEFINED,
            )
            return [p.mesh_path for p in proxies]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We were called from inside an existing loop (e.g. the pipeline).
                # Make a fresh loop in a thread to avoid "loop already running".
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    new_paths = ex.submit(asyncio.run, _run()).result()
            else:
                new_paths = loop.run_until_complete(_run())
        except RuntimeError:
            new_paths = asyncio.run(_run())

        info["available"] = any(p is not None for p in new_paths)
        return new_paths, info
    except Exception as e:  # pragma: no cover - exercised only when assets absent
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info


def apply_worst_match(state: dict, rank: int, seed: int) -> Tuple[dict, dict]:
    """
    Fork the saved scene state and re-run retrieval with the CLIP ranking inverted
    so each object's asset is swapped for the worst (lowest-CLIP) match.

    Returns (new_state, report). Degrades gracefully: when retrieval is
    unavailable (no HSSD DB / CLIP / OpenAI), the original assets are kept and
    the intent is recorded under `_worst_match` on the state (and
    per-object) so downstream tooling can see what was requested. Nothing crashes.
    """
    state = copy.deepcopy(state)
    fmt = detect_format(state)
    report = {
        "rank": rank, "seed": seed, "format": fmt,
        "applied": 0, "intended": 0, "available": False, "error": None,
    }

    if fmt == "hsm":
        objs, was_dict = _scene_objects_as_list(state["scene_objects"])
        new_paths, info = _try_worst_match_retrieval(objs, rank, seed)
        report["error"] = info.get("error")
        rebuilt = []
        for i, obj in enumerate(objs):
            report["intended"] += 1
            orig_path = obj.get("mesh_path")
            new_path = new_paths[i] if (new_paths and i < len(new_paths)) else None
            applied = new_path is not None and new_path != orig_path
            obj["_worst_match"] = {
                "rank": rank,
                "query": obj.get("name") or obj.get("obj_type") or "",
                "original_mesh_path": orig_path,
                "new_mesh_path": new_path if applied else None,
                "applied": applied,
            }
            if applied:
                obj["mesh_path"] = new_path
                report["applied"] += 1
            rebuilt.append(obj)
        if was_dict:
            state["scene_objects"] = {o["id"]: o for o in rebuilt}
        else:
            state["scene_objects"] = rebuilt
    else:
        objs = state["scene"]["object"]
        new_paths, info = _try_worst_match_retrieval(objs, rank, seed)
        report["error"] = info.get("error")
        for i, obj in enumerate(objs):
            report["intended"] += 1
            orig_model = obj.get("modelId", "")
            new_path = new_paths[i] if (new_paths and i < len(new_paths)) else None
            applied = new_path is not None
            obj["_worst_match"] = {
                "rank": rank,
                "modelId": orig_model,
                "new_mesh_path": new_path if applied else None,
                "applied": applied,
            }
            if applied:
                # modelId convention: "fpModel.<hssd_id>"; derive from mesh filename.
                import os
                hssd_id = os.path.splitext(os.path.basename(new_path))[0]
                obj["modelId"] = f"fpModel.{hssd_id}"
                report["applied"] += 1

    report["available"] = report["applied"] > 0
    state["_worst_match"] = report
    return state, report


# ---------------------------------------------------------------------------
# Substitution: within-category vs cross-category (lazy, graceful)
# ---------------------------------------------------------------------------


def _try_substitution_lookup(query: str, mode: str, seed: int) -> Optional[str]:
    """
    Lazy hook into hsm_core.retrieval to fetch a substitute asset id either in the
    SAME category (`mode == 'within'`, different instance) or a DIFFERENT category
    (`mode == 'cross'`). Returns the HSSD id, or None when retrieval is unavailable
    (no HSSD DB / CLIP / OpenAI). Pure-Python; all heavy imports are lazy.
    """
    try:  # pragma: no cover - exercised only with full assets present
        import numpy as np  # noqa: F401
        from hsm_core.retrieval import retrieve  # noqa: F401

        # A within/cross substitution needs the HSSD category index + embeddings +
        # CLIP + an OpenAI key, none of which are available here, so we never reach
        # a successful lookup in the audit environment.
        raise RuntimeError("substitution requires HSSD DB + CLIP + OpenAI")
    except Exception:
        return None


def apply_substitution(state: dict, mode: str, seed: int) -> Tuple[dict, dict]:
    """
    Attempt to replace each object's asset with a within- or cross-category
    substitute.

    Returns (new_state, report). Degrades gracefully: when the retrieval hook is
    unavailable, the original asset is kept and the intent is recorded as an
    {object_id: mode} map under `_substitution` on the state (and per-object)
    so downstream tooling can see what was requested. Handles both state formats.
    """
    if mode not in ("within", "cross"):
        raise ValueError(f"Unknown substitution mode: {mode!r} (expected 'within'/'cross')")
    state = copy.deepcopy(state)
    fmt = detect_format(state)
    intent: Dict[str, str] = {}
    report = {
        "mode": mode,
        "seed": seed,
        "format": fmt,
        "applied": 0,
        "intended": 0,
        "available": False,
        "intent": intent,
    }

    if fmt == "hsm":
        objs, was_dict = _scene_objects_as_list(state["scene_objects"])
        rebuilt = []
        for obj in objs:
            report["intended"] += 1
            obj_id = str(obj.get("id"))
            query = obj.get("name") or obj.get("obj_type") or ""
            new_id = _try_substitution_lookup(query, mode, seed)
            intent[obj_id] = mode
            obj["_substitution"] = {"mode": mode, "query": query, "applied": new_id is not None}
            if new_id is not None:
                report["applied"] += 1
            rebuilt.append(obj)
        if was_dict:
            state["scene_objects"] = {o["id"]: o for o in rebuilt}
        else:
            state["scene_objects"] = rebuilt
    else:
        for obj in state["scene"]["object"]:
            report["intended"] += 1
            obj_id = str(obj.get("id"))
            model_id = obj.get("modelId", "")
            new_id = _try_substitution_lookup(model_id, mode, seed)
            intent[obj_id] = mode
            obj["_substitution"] = {"mode": mode, "modelId": model_id, "applied": new_id is not None}
            if new_id is not None:
                report["applied"] += 1

    report["available"] = report["applied"] > 0
    state["_substitution"] = report
    return state, report


# ---------------------------------------------------------------------------
# IO / driver
# ---------------------------------------------------------------------------


def _load(scene_dir: Path) -> Tuple[Path, dict]:
    for name in ("hsm_scene_state.json", "stk_scene_state.json"):
        p = scene_dir / name
        if p.exists():
            with open(p) as f:
                return p, json.load(f)
    raise FileNotFoundError(f"No scene-state file under {scene_dir}")


def _write_variant(src: Path, state: dict, variant: str) -> Path:
    out = src.with_name(f"{src.stem}_{variant}{src.suffix}")
    with open(out, "w") as f:
        json.dump(state, f, indent=4)
    return out


def generate_all_variants(scene_dir: Path, seed: int = 42) -> Dict[str, str]:
    """Generate every removal + worst-match variant; return {variant: output_path}."""
    src, state = _load(scene_dir)
    written: Dict[str, str] = {}

    for variant, divisor in REMOVAL_DIVISORS.items():
        new_state = remove_objects(state, divisor, seed)
        written[variant] = str(_write_variant(src, new_state, variant))

    for variant, rank in WORST_MATCH_RANKS.items():
        new_state, _ = apply_worst_match(state, rank, seed)
        written[variant] = str(_write_variant(src, new_state, variant))

    for variant, mode in SUBST_MODES.items():
        new_state, _ = apply_substitution(state, mode, seed)
        written[variant] = str(_write_variant(src, new_state, variant))

    new_state = scramble_layout(state, seed)
    written["variant_scramble"] = str(_write_variant(src, new_state, "variant_scramble"))

    return written


# ---------------------------------------------------------------------------
# Named variant generation (user-facing scheme for cli.py --variants)
# ---------------------------------------------------------------------------


def build_variant(state: dict, family: str, seed: int, **kwargs) -> dict:
    """Build ONE variant state from a base state given a `family` and kwargs.

    Families: 'removal' (divisor), 'biggest' (), 'scramble' (scramble_rotation),
    'worst' (rank). All are PURE (deepcopy the input); 'worst' may invoke the
    lazy retrieval hook but degrades gracefully when assets are absent.
    """
    if family == "removal":
        return remove_objects(state, kwargs.get("divisor", 2), seed)
    if family == "biggest":
        return biggest_only(state)
    if family == "scramble":
        return scramble_layout(state, seed, scramble_rotation=kwargs.get("scramble_rotation", False))
    if family == "worst":
        new_state, _ = apply_worst_match(state, kwargs.get("rank", 0), seed)
        return new_state
    raise ValueError(f"Unknown variant family: {family!r}")


def _canonical_state_filename(state: dict) -> str:
    """The on-disk filename the renderer looks for, matching the state's format."""
    return "hsm_scene_state.json" if detect_format(state) == "hsm" else "stk_scene_state.json"


def generate_named_variants(
    scene_dir: Path,
    seed: int = 42,
    names: Optional[List[str]] = None,
    source_subdir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate the user-facing named variants, each in its OWN subfolder.

    Layout (matches cli.py's run-dir contract):

        <scene_dir>/                      <- the run dir (e.g. outputs/<stamp>/)
            <source_subdir>/              <- base scene lives here ("base")
                hsm_scene_state.json
            variant_01_half/
                hsm_scene_state.json      <- canonical name; renderer finds it
            variant_02_biggest-only/
                hsm_scene_state.json
            ...

    - ``scene_dir``: the run dir. Variant subfolders are created as its children.
    - ``source_subdir``: subfolder under ``scene_dir`` holding the base scene
      state (``"base"`` for cli.py). If None, the base state is read directly
      from ``scene_dir`` (so variant subfolders are also children of
      ``scene_dir`` — the original flat-source behavior the tests use).
    - Each variant is written under its own subfolder named after the variant,
      using the canonical state filename (``hsm_scene_state.json`` /
      ``stk_scene_state.json``) so ``render.py --scene-dir <that
      subfolder>`` resolves it and writes ``renderings/`` next to it.
    - A freshly loaded copy of the base state is used per variant so each is an
      independent fork (never compounds on another).

    Returns {variant_name: output_path}.
    """
    names = names or ALL_VARIANT_NAMES
    src_dir = scene_dir / source_subdir if source_subdir else scene_dir
    written: Dict[str, str] = {}
    for name in names:
        if name not in VARIANT_SPECS:
            raise ValueError(f"Unknown variant name: {name!r} (expected one of {ALL_VARIANT_NAMES})")
        family, kwargs = VARIANT_SPECS[name]
        # Re-load the base state each iteration so each variant is an independent fork
        # of the original (apply_worst_match/scramble/etc. deepcopy, but loading fresh
        # also protects against any in-place mutation of the loaded dict).
        _, base_state = _load(src_dir)
        new_state = build_variant(base_state, family, seed, **kwargs)

        variant_dir = scene_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        out = variant_dir / _canonical_state_filename(new_state)
        with open(out, "w") as f:
            json.dump(new_state, f, indent=4)
        written[name] = str(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate content variants for HSM scenes.")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    written = generate_all_variants(args.scene_dir, args.seed)
    for variant, path in written.items():
        print(f"[scene_variants] {variant} -> {path}")


if __name__ == "__main__":
    main()
