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

# Resolutions (px) — paper Table 1.
RESOLUTIONS = [196, 224, 256, 336, 384, 448, 512, 768, 1024]
# Focal lengths (mm) — paper Table 1.
FOCAL_LENGTHS = [16, 24, 35, 50, 85, 100, 200]
# Background gray levels (0..255) — paper Table 1 (6 grays).
BACKGROUND_GRAYS = [0, 65, 128, 186, 204, 255]
# Chromatic backgrounds (R, G, B) — paper Table 1.
BACKGROUND_CHROMATIC: list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# Floor-texture background sentinel — documented as a level but NOT rendered by this
# harness (textured floor compositing is out of scope; recorded for completeness).
FLOOR_TEXTURE_BACKGROUND = "floor_texture"
HDRIS = ["city", "courtyard", "forest", "interior", "night", "studio", "sunrise", "sunset"]

# Camera pitch / yaw (degrees). In the bpa convention used by Renderer.render_perspective,
# rotation=(pitch, 0, yaw); pitch == 0 looks straight down (top-down).
PITCHES = [0, 15, 30, 45, 60, 75, 90]
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]

# ---------------------------------------------------------------------------
# Baseline / reference configuration
# ---------------------------------------------------------------------------

BASELINE_RESOLUTION = 512
BASELINE_FOCAL_LENGTH = 50
BASELINE_BACKGROUND: Tuple[int, int, int] = (128, 128, 128)
BASELINE_HDRI = "city"
BASELINE_PITCH = 0
BASELINE_YAW = 0
# Baseline yaw used when sweeping pitch alone; baseline pitch used when sweeping yaw alone.
BASELINE_YAW_PITCH = 45


def _gray(level: int) -> Tuple[int, int, int]:
    return (level, level, level)


# Canonical ordered list of audit phases (used by drivers / CLIs).
ALL_PHASES = ["1a", "1b", "1b_chroma", "1c", "1d", "2", "2_pitch", "2_yaw"]


def phase_levels(phase: str) -> dict:
    """
    Return the sweep specification for an audit phase.

    Each returned dict carries the *baseline* for every factor plus a `vary` key
    naming the factor being swept and a `levels` list of the values to render.

    Phases:
        '1a'        — resolution sweep
        '1b'        — background-gray sweep
        '1b_chroma' — chromatic-background sweep (R/G/B)
        '1c'        — HDRI sweep
        '1d'        — focal-length sweep
        '2'         — pitch x yaw camera-angle grid (backward-compat)
        '2_pitch'   — pitch sweep with yaw fixed at BASELINE_YAW (0)
        '2_yaw'     — yaw sweep with pitch fixed at BASELINE_YAW_PITCH (45)
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
    if phase == "1b_chroma":
        return {**base, "vary": "background", "levels": list(BACKGROUND_CHROMATIC)}
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
    if phase == "2_pitch":
        # Pitch sweep with yaw fixed at the baseline yaw (0).
        return {**base, "yaw": BASELINE_YAW, "vary": "pitch", "levels": list(PITCHES)}
    if phase == "2_yaw":
        # Yaw sweep with pitch fixed at BASELINE_YAW_PITCH (45).
        return {**base, "pitch": BASELINE_YAW_PITCH, "vary": "yaw", "levels": list(YAWS)}
    raise ValueError(
        f"Unknown phase: {phase!r} (expected one of {', '.join(ALL_PHASES)})"
    )
