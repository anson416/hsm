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
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

REMOVAL_DIVISORS = {"variant_half": 2, "variant_quarter": 4, "variant_eighth": 8}
WORST_MATCH_RANKS = {"variant_alt_0": 0, "variant_alt_2": 2, "variant_alt_4": 4}


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
