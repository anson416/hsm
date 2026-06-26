"""
vlmunr_config.py — Canonical factor levels for the VLM-unreliability audit harness.

Self-contained: no external `vlmunr` import. Defines the resolution / focal-length /
background-gray / HDRI / camera-angle sweeps, the baseline (reference) configuration,
and a `phase_levels()` helper that maps an audit phase id to the factor it varies.
"""

from __future__ import annotations

from typing import Tuple

# ---------------------------------------------------------------------------
# Factor levels
# ---------------------------------------------------------------------------

RESOLUTIONS = [224, 256, 384, 448, 512, 640, 768, 1024]
FOCAL_LENGTHS = [24, 35, 50, 85, 100, 200]
BACKGROUND_GRAYS = [0, 18, 65, 117, 128, 186, 204, 255]
HDRIS = ["city", "courtyard", "forest", "interior", "night", "studio", "sunrise", "sunset"]

# Camera pitch / yaw (degrees). In the bpa convention used by Renderer.render_perspective,
# rotation=(pitch, 0, yaw); pitch == 0 looks straight down (top-down).
PITCHES = [0, 30, 60, 90]
YAWS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# ---------------------------------------------------------------------------
# Baseline / reference configuration
# ---------------------------------------------------------------------------

BASELINE_RESOLUTION = 512
BASELINE_FOCAL_LENGTH = 50
BASELINE_BACKGROUND: Tuple[int, int, int] = (128, 128, 128)
BASELINE_HDRI = "city"
BASELINE_PITCH = 0
BASELINE_YAW = 0


def _gray(level: int) -> Tuple[int, int, int]:
    return (level, level, level)


def phase_levels(phase: str) -> dict:
    """
    Return the sweep specification for an audit phase.

    Each returned dict carries the *baseline* for every factor plus a `vary` key
    naming the factor being swept and a `levels` list of the values to render.

    Phases:
        '1a' — resolution sweep
        '1b' — background-gray sweep
        '1c' — HDRI sweep
        '1d' — focal-length sweep
        '2'  — pitch x yaw camera-angle grid
    """
    phase = phase.lower()
    base = {
        "resolution": BASELINE_RESOLUTION,
        "focal_length": BASELINE_FOCAL_LENGTH,
        "background": BASELINE_BACKGROUND,
        "hdri": BASELINE_HDRI,
        "pitch": BASELINE_PITCH,
        "yaw": BASELINE_YAW,
    }

    if phase == "1a":
        return {**base, "vary": "resolution", "levels": list(RESOLUTIONS)}
    if phase == "1b":
        return {**base, "vary": "background", "levels": [_gray(g) for g in BACKGROUND_GRAYS]}
    if phase == "1c":
        return {**base, "vary": "hdri", "levels": list(HDRIS)}
    if phase == "1d":
        return {**base, "vary": "focal_length", "levels": list(FOCAL_LENGTHS)}
    if phase == "2":
        return {
            **base,
            "vary": "pitch_yaw",
            "levels": [(p, y) for p in PITCHES for y in YAWS],
        }
    raise ValueError(f"Unknown phase: {phase!r} (expected one of 1a, 1b, 1c, 1d, 2)")
