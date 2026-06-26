"""
vlmunr_variants.py — Content-variant generators for the VLM audit harness.

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

import vlmunr_render as rnd

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

REMOVAL_DIVISORS = {"variant_half": 2, "variant_quarter": 4, "variant_eighth": 8}
WORST_MATCH_RANKS = {"variant_alt_0": 0, "variant_alt_2": 2, "variant_alt_4": 4}
# Substitution modes: same category (different instance) vs different category.
SUBST_MODES = {"variant_subst_within": "within", "variant_subst_cross": "cross"}


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
            xs.append(-float(p[0]))
            zs.append(float(p[2]))
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


def scramble_stk(state: dict, seed: int) -> dict:
    """
    PURE: relocate every stk object to a random (x, z) within the room footprint,
    preserving the object set, ids/indices and rotation/orientation. Only the
    translation components of each column-major transform are changed; the 3x3
    rotation block (orientation) is left intact by re-encoding with the decoded
    rotation. Reuses decode/encode helpers so the column-major + Y-up convention
    matches the renderer. Deterministic for a fixed seed.
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
        obj["transform"]["data"] = rnd.encode_stk_transform(new_pos, rot_deg)
    return state


def scramble_hsm(state: dict, seed: int) -> dict:
    """
    PURE: relocate every hsm scene_object to a random (x, z) within the room
    footprint, keeping y (height) and rotation. Bounds derive from object position
    extents (hsm_scene_state has no arch). Preserves dict/list shape and ids.
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
    if was_dict:
        state["scene_objects"] = {o["id"]: o for o in objs}
    else:
        state["scene_objects"] = objs
    return state


def scramble_layout(state: dict, seed: int) -> dict:
    """Dispatch layout scramble on detected format."""
    fmt = detect_format(state)
    if fmt == "stk":
        return scramble_stk(state, seed)
    return scramble_hsm(state, seed)


# ---------------------------------------------------------------------------
# Worst-match retrieval hook (lazy, graceful)
# ---------------------------------------------------------------------------


def _try_worst_match_lookup(
    query: str, rank: int
) -> Optional[str]:
    """
    Lazy hook into hsm_core.retrieval to fetch the WORST (lowest-CLIP) match at the
    given ascending rank. Returns the HSSD id, or None if retrieval is unavailable
    (no HSSD DB / CLIP / OpenAI). Pure-Python; all heavy imports are lazy.
    """
    try:  # pragma: no cover - exercised only with full assets present
        import numpy as np  # noqa: F401
        from hsm_core.retrieval import retrieve  # noqa: F401

        # Real retrieval ranks best-first via (-sim).argsort(); worst-match is the
        # ascending argsort. The concrete call needs the HSSD embeddings index +
        # CLIP + an OpenAI key, none of which are available here, so we never reach
        # a successful lookup in the audit environment.
        raise RuntimeError("retrieval requires HSSD DB + CLIP + OpenAI")
    except Exception:
        return None


def apply_worst_match(state: dict, rank: int, seed: int) -> Tuple[dict, dict]:
    """
    Attempt to replace each object's asset with a worst-match at `rank`.

    Returns (new_state, report). Degrades gracefully: when the retrieval hook is
    unavailable, the original asset is kept and the intent is recorded under
    `_vlmunr_worst_match` on the state (and per-object) so downstream tooling can
    see what was requested.
    """
    state = copy.deepcopy(state)
    fmt = detect_format(state)
    report = {"rank": rank, "seed": seed, "format": fmt, "applied": 0, "intended": 0, "available": False}

    if fmt == "hsm":
        objs, was_dict = _scene_objects_as_list(state["scene_objects"])
        rebuilt = []
        for obj in objs:
            report["intended"] += 1
            query = obj.get("name") or obj.get("obj_type") or ""
            new_id = _try_worst_match_lookup(query, rank)
            obj.setdefault("_vlmunr_worst_match", {})
            obj["_vlmunr_worst_match"] = {"rank": rank, "query": query, "applied": new_id is not None}
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
            model_id = obj.get("modelId", "")
            new_id = _try_worst_match_lookup(model_id, rank)
            obj["_vlmunr_worst_match"] = {"rank": rank, "modelId": model_id, "applied": new_id is not None}
            if new_id is not None:
                report["applied"] += 1

    report["available"] = report["applied"] > 0
    state["_vlmunr_worst_match"] = report
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
    {object_id: mode} map under `_vlmunr_substitution` on the state (and per-object)
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
            obj["_vlmunr_substitution"] = {"mode": mode, "query": query, "applied": new_id is not None}
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
            obj["_vlmunr_substitution"] = {"mode": mode, "modelId": model_id, "applied": new_id is not None}
            if new_id is not None:
                report["applied"] += 1

    report["available"] = report["applied"] > 0
    state["_vlmunr_substitution"] = report
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate content variants for HSM scenes.")
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    written = generate_all_variants(args.scene_dir, args.seed)
    for variant, path in written.items():
        print(f"[vlmunr_variants] {variant} -> {path}")


if __name__ == "__main__":
    main()
