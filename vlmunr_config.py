"""
vlmunr_config.py — Render-factor levels + sweep definitions for the VLM-unreliability
audit harness.

Self-contained (no external `vlmunr` import). Defines the resolution / focal-length /
background / HDRI / camera-angle sweeps, the single-reference baseline, and producers
that turn a request (`--render` single config, or `--render-all` six sweeps) into a
DEDUPED list of render configs. Each render config is one transparent master to render
plus the list of background colors to composite over it.

Filename contract (see vlmunr_render.py):
  transparent master : render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}.png
  composited (bg)   : render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}_bg-{r}-{g}-{b}.png
"""

from __future__ import annotations

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Factor levels
# ---------------------------------------------------------------------------

# Resolutions (px) — paper Table 1.
RESOLUTIONS: List[int] = [196, 224, 256, 336, 384, 448, 512, 768, 1024]
# Focal lengths (mm) — paper Table 1.
FOCAL_LENGTHS: List[int] = [16, 24, 35, 50, 85, 100, 200]
# Background colors (R,G,B) — paper Table 1 grays + chromatic, incl. the new 118 gray.
BACKGROUNDS: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (65, 65, 65),
    (118, 118, 118),
    (128, 128, 128),
    (186, 186, 186),
    (204, 204, 204),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]
HDRIS: List[str] = ["city", "courtyard", "forest", "interior", "night", "studio", "sunrise", "sunset"]

# Camera pitch / yaw (degrees). In the bpa convention used by
# Renderer.render_perspective, rotation=(pitch, 0, yaw); pitch == 0 looks straight
# down (top-down). The filename always carries this same value (pitch 0 == top-down),
# so filenames are consistent across methods regardless of any per-method convention.
PITCHES: List[int] = [0, 15, 30, 45, 60, 75, 90]
YAWS: List[int] = [0, 45, 90, 135, 180, 225, 270, 315]

# ---------------------------------------------------------------------------
# Baseline / reference configuration (the single `--render` config)
# ---------------------------------------------------------------------------

BASELINE_RESOLUTION = 512
BASELINE_FOCAL_LENGTH = 50
BASELINE_BACKGROUND: Tuple[int, int, int] = (255, 255, 255)
BASELINE_HDRI = "city"
BASELINE_PITCH = 0
BASELINE_YAW = 0

# When sweeping yaw alone, pitch is fixed at 45 (NOT the baseline 0) so the yaw
# rotation is actually visible (a top-down view is rotation-invariant about yaw).
BASELINE_YAW_PITCH = 45


class RenderConfig:
    """One transparent master to render, plus the bg colors to composite over it.

    A transparent master is fully determined by (resolution, focal_length, pitch,
    yaw, hdri). The same camera+env config can be composited over any number of
    background colors without re-rendering, so multiple background requests collapse
    onto a single RenderConfig.
    """

    __slots__ = ("resolution", "focal_length", "pitch", "yaw", "hdri", "backgrounds")

    def __init__(
        self,
        resolution: int,
        focal_length: int,
        pitch: int,
        yaw: int,
        hdri: str,
        backgrounds: List[Tuple[int, int, int]],
    ) -> None:
        self.resolution = resolution
        self.focal_length = focal_length
        self.pitch = pitch
        self.yaw = yaw
        self.hdri = hdri
        self.backgrounds = list(backgrounds)

    @property
    def key(self) -> Tuple[int, int, int, int, str]:
        """Camera+env identity (transparent master dedup key)."""
        return (self.resolution, self.focal_length, self.pitch, self.yaw, self.hdri)

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return (
            f"RenderConfig(res={self.resolution}, focal={self.focal_length}, "
            f"pitch={self.pitch}, yaw={self.yaw}, env={self.hdri}, "
            f"bgs={self.backgrounds})"
        )


# ---------------------------------------------------------------------------
# Sweep producers
# ---------------------------------------------------------------------------

def single_render_config() -> RenderConfig:
    """The one config rendered by `--render`: baseline (white bg), top-down, city."""
    return RenderConfig(
        resolution=BASELINE_RESOLUTION,
        focal_length=BASELINE_FOCAL_LENGTH,
        pitch=BASELINE_PITCH,
        yaw=BASELINE_YAW,
        hdri=BASELINE_HDRI,
        backgrounds=[BASELINE_BACKGROUND],
    )


def _render_all_specs() -> List[RenderConfig]:
    """The six sweeps, expanded to per-config RenderConfigs (NOT yet deduped).

    Per spec:
      (1) resolution sweep      — vary res,   bg white, focal 50, pitch 0, yaw 0, city
      (2) focal-length sweep    — vary focal, res 512, bg white, pitch 0, yaw 0, city
      (3) pitch sweep           — vary pitch, res 512, bg white, focal 50, yaw 0, city
      (4) yaw sweep             — vary yaw,   res 512, bg white, focal 50, pitch 45, city
      (5) env sweep             — vary env,   res 512, bg white, focal 50, pitch 0, yaw 0
      (6) background sweep      — vary bg,    res 512, focal 50, pitch 0, yaw 0, city
    """
    specs: List[RenderConfig] = []
    white = [BASELINE_BACKGROUND]

    # (1) resolution
    for res in RESOLUTIONS:
        specs.append(RenderConfig(res, 50, 0, 0, "city", white))
    # (2) focal length
    for focal in FOCAL_LENGTHS:
        specs.append(RenderConfig(512, focal, 0, 0, "city", white))
    # (3) pitch
    for pitch in PITCHES:
        specs.append(RenderConfig(512, 50, pitch, 0, "city", white))
    # (4) yaw (pitch fixed at 45)
    for yaw in YAWS:
        specs.append(RenderConfig(512, 50, BASELINE_YAW_PITCH, yaw, "city", white))
    # (5) env map
    for env in HDRIS:
        specs.append(RenderConfig(512, 50, 0, 0, env, white))
    # (6) background — one camera config, every bg color composited over it
    specs.append(RenderConfig(512, 50, 0, 0, "city", list(BACKGROUNDS)))
    return specs


def render_all_configs() -> List[RenderConfig]:
    """The deduped set of render configs for `--render-all`.

    Collapses all six sweeps onto their (res, focal, pitch, yaw, env) identity: a
    camera+env config rendered by one sweep is the same transparent master as the
    same config rendered by another sweep, so it is rendered once and composited
    over the UNION of every background color requested for it across sweeps.

    The baseline (512, 50, 0, 0, city) is requested by every sweep and by the bg
    sweep (which adds all 10 colors), so it ends up with all 10 backgrounds.
    """
    merged: dict = {}
    for spec in _render_all_specs():
        k = spec.key
        if k in merged:
            seen = merged[k]
            for bg in spec.backgrounds:
                if bg not in seen.backgrounds:
                    seen.backgrounds.append(bg)
        else:
            merged[k] = RenderConfig(
                spec.resolution,
                spec.focal_length,
                spec.pitch,
                spec.yaw,
                spec.hdri,
                list(spec.backgrounds),
            )
    return list(merged.values())


# Canonical mode names (drives the renderer CLI).
SINGLE_MODE = "single"
ALL_MODE = "all"
ALL_MODES = [SINGLE_MODE, ALL_MODE]
