#!/usr/bin/env python3
"""
cli.py — Single-command HSM scene generation + content-variant driver.

Generate a realistic 3D indoor scene from a textual description, and (optionally)
write four content variants derived from the base scene WITHOUT regenerating the
scene or re-calling the LLM:

    variant_01_half          keep round(n/2) objects (seeded sample)
    variant_02_biggest-only  keep the single largest object (by bbox volume)
    variant_03_scrambled     randomize every object's position AND heading in-room
    variant_04_worst-object  fork + re-run retrieval with the CLIP ranking inverted
                             so each object's 3D asset is swapped for the worst match

The LLM connection (--base-url/--api-key/--model/--temperature) is wired through
the OPENAI_* env vars, which hsm_core.vlm.gpt.Session reads at construction.

Output layout (per run), under ./outputs/ :

    outputs/<YYYYMMDD-HHMMSS>/
        config.json                       prompt + all CLI/LLM/data configs
        base/                             the generated BASE scene
            room_scene.glb                base scene GLB
            hsm_scene_state.json          base scene state (mesh_path + pose)
            stk_scene_state.json          base scene state (column-major STK)
            scene.log                     run log
            visualizations/ ...           stage visualizations
            scene_motifs/ ...             motif GLBs/pickles
            renderings/ ...               (only if --render / --render-all)
        variant_01_half/                  (only with --variants)
            hsm_scene_state.json          round(n/2) objects kept
            renderings/ ...               (only if --render / --render-all)
        variant_02_biggest-only/
            hsm_scene_state.json          single largest object kept
        variant_03_scrambled/
            hsm_scene_state.json          positions + headings randomized in-room
        variant_04_worst-object/
            hsm_scene_state.json          assets swapped for worst-CLIP matches

Rendering (--render / --render-all):
  --render      one baseline render per scene subfolder (512px, white bg, 50mm,
                pitch 0 = top-down, yaw 0, city env). Writes a transparent master
                AND a white-composited PNG into <subfolder>/renderings/.
  --render-all  the full six-factor sweep (resolution / focal / pitch / yaw /
                env / background), deduped per camera config, into renderings/.
  Either flag, with --prompt, renders the generated base (+ variants if
                --variants). Either flag, with --path outputs/<stamp>/, skips
                generation and renders base/ + every variant_* already present.
  --prompt and --path are mutually exclusive; --render and --render-all are
  mutually exclusive.

  Filenames (under <subfolder>/renderings/):
    transparent master : render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}.png
    composited (bg)    : render_res-{R}_focal-{F}_pitch-{P}_yaw-{Y}_env-{ENV}_bg-{r}-{g}-{b}.png
  Rendering always uses bpa's two-stage recipe (transparent master first, env-map
  lit; then composite the bg color via PIL) and fit_ratio=1 (tight-fit). Walls
  use the dollhouse convention: camera-facing wall faces are transparent (back-
  face culling by surface normal); far walls + their door/window openings remain
  visible. Rendering runs IN-PROCESS under the `hsm` conda env, which carries
  bpy (install via the `[render]` extra). There is no separate render env and no
  subprocess: cli.py imports `render` lazily and calls render_configs() directly,
  so bpy is only loaded when rendering actually runs.

Each variant subfolder holds a standalone scene-state file under the canonical
name (hsm_scene_state.json / stk_scene_state.json) the renderer looks for, so
you can render any variant on its own:

    conda activate hsm          # bpy lives here too
    python render.py --scene-dir outputs/<stamp>/variant_03_scrambled
    # -> writes outputs/<stamp>/variant_03_scrambled/renderings/

Usage:
    conda activate hsm
    # generate (+ variants) and render the baseline of each scene:
    python cli.py --prompt "A cozy bedroom with a bed and nightstand." \
        --base-url https://api.openai.com/v1 --api-key sk-... \
        --model gpt-4o-2024-08-06 --temperature 0.7 --variants --render
    # render the full sweep of an existing run (no generation):
    python cli.py --path outputs/20260708-023434 --render-all

==============================================================================
EXTERNAL RESOURCES THIS METHOD REQUIRES (and how to prepare them)
==============================================================================

HSM is NOT a self-contained generator: it needs (a) an LLM API for scene
decomposition/arrangement, (b) a local CLIP model for object retrieval, and
(c) several local datasets/databases of 3D assets + precomputed indices. If any
of these are missing the run will fail (or, for the worst-object variant, degrade
gracefully). Below is the complete list and step-by-step preparation.

NOTE: the bundled `./setup.sh` automates most of the dataset downloads. This
section documents what it does and how to do it manually / point the CLI at
already-downloaded data on another disk.

--------------------------------------------------------------------
1. LLM API (OpenAI-compatible)  — REQUIRED for generation
--------------------------------------------------------------------
HSM uses a GPT-class VLM to decompose the room description into objects, generate
arrangement code, and assign WordNet synset keys. It calls the OpenAI chat
completions API (any OpenAI-compatible endpoint works).

  What you need:
    - An API key.
    - A base URL (official OpenAI: https://api.openai.com/v1 ; or a compatible
      proxy/gateway).
    - A model name. Defaults baked into hsm_core.vlm.gpt: "gpt-5.1-2025-11-13".
      Common alternatives: "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14".

  How to provide it (any of):
    a) CLI flags:  --api-key sk-... --base-url https://api.openai.com/v1 \
                      --model gpt-4o-2024-08-06 --temperature 0.7
    b) Environment variables (the CLI exports these for the pipeline):
         OPENAI_API_KEY, OPENAI_BASE_URL,
         OPENAI_MODEL, OPENAI_TEMPERATURE
    c) A `.env` file in the repo root with OPENAI_API_KEY=sk-...  (gpt.py calls
       load_dotenv(); OPENAI_API_KEY is read as a fallback for the API key).

  Get an OpenAI key: https://platform.openai.com/api-keys

--------------------------------------------------------------------
2. Hugging Face token  — REQUIRED only to DOWNLOAD the asset datasets (below)
--------------------------------------------------------------------
The 3D asset databases are hosted on Hugging Face and require license acceptance.
You need a HF token once, to download them; after that the token is not needed
to run generation.

  Get a token: https://huggingface.co/settings/tokens
  Accept the HSSD license: https://huggingface.co/datasets/hssd/hssd-models
  Provide it:  export HF_TOKEN=hf_...   (or put HF_TOKEN=... in .env)
  Then log in: conda activate hsm && hf auth login --token $HF_TOKEN

--------------------------------------------------------------------
3. HSSD 3D models (the object asset database)  — REQUIRED
   ~72 GB, under data/hssd-models/  (override with --hssd-dir)
--------------------------------------------------------------------
Every placed object's 3D mesh (.glb) is retrieved from the Habitat Synthetic
Scenes Dataset (HSSD). This is the big one. Two parts:

  a) Full HSSD models (~72 GB):
       cd data
       git lfs install
       git clone https://huggingface.co/datasets/hssd/hssd-models
     (or:  hf download hssd/hssd-models --repo-type=dataset --local-dir data/hssd-models)

  b) Decomposed part models (furniture broken into parts):
       hf download hssd/hssd-hab --repo-type=dataset \
           --include "objects/decomposed/**/*_part_*.glb" \
           --exclude "objects/decomposed/**/*_part.*.glb" \
           --local-dir "data/hssd-models"

  After download, data/hssd-models/ should contain:
      objects/<id[0]>/<id>.glb          (whole-object meshes)
      objects/decomposed/<base>/...     (part meshes)
      support-surfaces/<id>/...         (see #5 below)

--------------------------------------------------------------------
4. Preprocessed retrieval data  — REQUIRED
   under data/preprocessed/  (override with --data-dir)
--------------------------------------------------------------------
Precomputed CLIP embeddings + indices that map object labels -> candidate HSSD
meshes via WordNet synset keys. Without these, retrieval cannot rank/select
assets. Files:

      data/preprocessed/clip_hssd_embeddings.npy          (CLIP image embeddings of all HSSD meshes)
      data/preprocessed/clip_hssd_embeddings_index.yaml    (mesh-id index for the .npy)
      data/preprocessed/hssd_wnsynsetkey_index.json        (WordNet synset -> mesh ids)
      data/preprocessed/object_categories.json             (mesh-id -> category/label)

  How to get them: download `data.zip` from the latest HSM release and unzip it
  at the repo root — it creates data/motif_library/ and data/preprocessed/:
      https://github.com/3dlg-hcvc/hsm/releases   (grab data.zip)
      unzip data.zip        # run from repo root

--------------------------------------------------------------------
5. Support-surface data  — REQUIRED (for placing small objects on furniture)
   under data/hssd-models/support-surfaces/  (override with --hssd-dir)
--------------------------------------------------------------------
Precomputed support regions (table tops, shelf surfaces) used to place small
objects on top of furniture.

  How to get them: download `support-surfaces.zip` from the latest HSM release
  and move it under data/hssd-models/:
      https://github.com/3dlg-hcvc/hsm/releases   (grab support-surfaces.zip)
      unzip support-surfaces.zip -d data/hssd-models/

--------------------------------------------------------------------
6. Motif library (meta-programs)  — REQUIRED (unless --skip-scene-motifs)
   under data/motif_library/meta_programs/  (override with --data-dir)
--------------------------------------------------------------------
Learned meta-program JSON files (in_front_of.json, etc.) that define how object
arrangements are composed. These ship inside data.zip from #4.

--------------------------------------------------------------------
7. CLIP model  — REQUIRED (auto-downloaded on first use)
--------------------------------------------------------------------
HSM loads a local CLIP model (model_manager.ModelManager.get_clip_model_async)
to compute text<->mesh similarities during retrieval. The weights are cached by
HuggingFace/torch on first use (~600 MB) — needs internet the first time only.

--------------------------------------------------------------------
8. Blender (bpy)  — NOT needed for generation; only for rendering
--------------------------------------------------------------------
Generation itself does NOT use Blender. Rendering (render.py) needs `bpy`, which
now lives in the SAME `hsm` conda env (install via the `[render]` extra:
`pip install -e ".[render]"`). bpy + torch coexist in one env, so cli.py renders
in-process — there is no separate render env. A generate-only install (without
the `[render]` extra) skips bpy and just leaves the `--render*` flags inert until
bpy is installed.

==============================================================================
QUICKSTART (full manual prep from scratch)
==============================================================================
    # 0. env
    conda activate hsm
    cp .env.example .env && vim .env        # set OPENAI_API_KEY and HF_TOKEN
    hf auth login --token "$HF_TOKEN"

    # 1. datasets (the big one is the ~72GB HSSD clone)
    cd data && git lfs install
    git clone https://huggingface.co/datasets/hssd/hssd-models && cd ..
    hf download hssd/hssd-hab --repo-type=dataset \
        --include "objects/decomposed/**/*_part_*.glb" \
        --exclude "objects/decomposed/**/*_part.*.glb" \
        --local-dir "data/hssd-models"
    # from https://github.com/3dlg-hcvc/hsm/releases : grab data.zip + support-surfaces.zip
    unzip data.zip                          # -> data/preprocessed, data/motif_library
    unzip support-surfaces.zip -d data/hssd-models

    # 2. generate
    python cli.py --prompt "A small living room with a sofa and coffee table." \
        --model gpt-4o-2024-08-06 --variants

    # (or, if your data lives elsewhere on another disk, override the paths):
    python cli.py --prompt "..." --variants \
        --hssd-dir /mnt/bigdisk/hssd-models --data-dir /mnt/bigdisk/hsm-data

==============================================================================
ENVIRONMENT NOTES
==============================================================================
- Run under the `hsm` conda env (it has openai, torch, omegaconf, trimesh, clip,
  python-dotenv, AND bpy — install the `[render]` extra for rendering). There is
  no separate render env; generation and rendering both run here.
- If --hssd-dir / --data-dir are omitted, the defaults data/hssd-models and
  data/ (relative to this repo) are used, matching ./setup.sh's layout.

==============================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data-path overrides (--hssd-dir / --data-dir)
# ---------------------------------------------------------------------------


def _apply_data_paths(hssd_dir: str | None, data_dir: str | None) -> tuple[Path, Path]:
    """Override HSM's dataset locations before the retrieval/support modules use them.

    HSM captures dataset paths at import time into module-level constants derived
    from hsm_core.config (HSSD_PATH, DATA_PATH) and a few other modules
    (SUPPORT_DIR, EMBEDDING_PATH, LIB_META_PROGRAMS_DIR). To override from the CLI
    we patch hsm_core.config first, then fix up the constants that were already
    captured. Must run BEFORE hsm_core.scene.processing is imported so the
    transitive imports see the patched values where they read at call time.

    Returns the resolved (hssd_path, data_path) for logging/config.
    """
    import hsm_core.config as hcfg

    if hssd_dir:
        hcfg.HSSD_PATH = Path(hssd_dir).resolve()
    if data_dir:
        hcfg.DATA_PATH = Path(data_dir).resolve()
    hssd_path = hcfg.HSSD_PATH
    data_path = hcfg.DATA_PATH

    # Fix up constants captured at import time in other modules. These are read at
    # call time from the module attribute, so reassigning the module global works.
    try:
        import hsm_core.support_region.loader as _sl

        _sl.SUPPORT_DIR = hssd_path / "support-surfaces"
    except Exception:
        pass
    try:
        import hsm_core.retrieval.model.embeddings as _emb

        _emb.EMBEDDING_PATH = data_path / "preprocessed"
    except Exception:
        pass
    try:
        import hsm_core.scene_motif.utils.library as _lib

        _lib.LIB_META_PROGRAMS_DIR = data_path / "motif_library" / "meta_programs"
    except Exception:
        pass
    return hssd_path, data_path


# IMPORTANT: import the pipeline AFTER _apply_data_paths can run. We defer the
# heavy imports to main() so --hssd-dir/--data-dir take effect first. These names
# are imported lazily inside _build_config / _run_pipeline below.


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> DictConfig | ListConfig:
    """Build the HSM config from scene_config.yaml + CLI overrides."""
    from argparser import HSMArgumentParser

    parser = HSMArgumentParser(PROJECT_ROOT)
    ns = argparse.Namespace(
        desc=args.prompt,
        output="outputs",  # the real run dir is set via output_dir_override below
        types=["large", "wall", "ceiling", "small"],
        extra_types=["large", "wall", "ceiling", "small"],
        skip_scene_motifs=False,
        skip_solver=False,
        skip_spatial_optimization=False,
    )
    cfg = parser.get_config(ns)

    # Inject an `llm` block so model-type routing picks 'gpt'. The actual model
    # name / temperature reach the Session via OPENAI_* env vars (set in
    # _apply_llm_env), which every create_session() site reads uniformly.
    cfg.llm = OmegaConf.create({"model_type": "gpt", "model_name": args.model})
    cfg.execution.result_dir = str(Path(args.output).resolve())
    return cfg


def _apply_llm_env(args: argparse.Namespace) -> None:
    """Export OPENAI_* env vars consumed by hsm_core.vlm.gpt.Session."""
    if args.base_url is not None:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key is not None:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.model is not None:
        os.environ["OPENAI_MODEL"] = args.model
    if args.temperature is not None:
        os.environ["OPENAI_TEMPERATURE"] = str(args.temperature)


def _write_config_json(
    run_dir: Path,
    args: argparse.Namespace,
    cfg: DictConfig | ListConfig,
    hssd_path: Path,
    data_path: Path,
) -> Path:
    """Persist the prompt + all configs (incl. data paths) for reproducibility."""
    config_path = run_dir / "config.json"
    has_key = bool(args.api_key or os.environ.get("OPENAI_API_KEY"))
    payload = {
        "prompt": args.prompt,
        "llm": {
            "base_url": args.base_url or os.environ.get("OPENAI_BASE_URL"),
            "model": args.model or os.environ.get("OPENAI_MODEL"),
            "temperature": args.temperature if args.temperature is not None else 0.7,
            "api_key": "<redacted>" if has_key else None,
        },
        "data": {
            "hssd_dir": str(hssd_path),
            "data_dir": str(data_path),
        },
        "variants": args.variants,
        "variant_seed": args.seed,
        "render": args.render,
        "render_all": args.render_all,
        "render_mode": "all"
        if args.render_all
        else ("single" if args.render else None),
        "hsm_config": OmegaConf.to_container(cfg, resolve=True),
        "run_dir": str(run_dir),
    }
    with open(config_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return config_path


# ---------------------------------------------------------------------------
# Pipeline driving (mirrors main.process_scene but captures the in-memory Scene)
# ---------------------------------------------------------------------------


async def _run_pipeline(cfg: DictConfig | ListConfig, run_dir: Path, logger) -> dict:
    """Run setup + stages + cleanup, returning the final context (holds the Scene)."""
    from hsm_core.scene.processing import (
        create_processing_pipeline,
        process_cleanup_stage,
        setup_scene_generation,
    )

    start_time = time.time()
    logger.info("Starting scene generation pipeline (cli)")

    # All base-scene artifacts (room_scene.glb, hsm/stk_scene_state.json,
    # visualizations/, scene_motifs/, scene.log) go under <run_dir>/base/ so the
    # run dir has a clean base/ + variant_*/ + config.json layout.
    base_dir = run_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    context = await setup_scene_generation(
        cfg,
        project_root=PROJECT_ROOT,
        output_dir_override=base_dir,  # write everything under outputs/<timestamp>/base/
        timestamp=False,  # base_dir IS the dir; no extra timestamp subdir
    )
    context["start_time"] = start_time
    pipeline = create_processing_pipeline(cfg, context["is_loaded_scene"])
    logger.info(f"Pipeline created with {len(pipeline)} stages")

    for stage_name, stage_func in pipeline:
        logger.info(f"Stage: {stage_name}")
        context = await stage_func(context, cfg)

    # The pipeline saved once with save_scene_state=False (default). Re-save WITH
    # save_scene_state=True so hsm_scene_state.json is emitted under base/ — the
    # variants layer and renderer prefer it (carries mesh_path + pose, no HSSD_DIR
    # needed).
    scene = context["scene"]
    out_dir = context["output_dir_override"]
    scene.save(out_dir, recreate_scene=True, save_scene_state=True)
    logger.info(f"Re-saved scene (with scene state) to {out_dir}")

    await process_cleanup_stage(context, cfg)
    return context


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------


def _generate_variants(run_dir: Path, seed: int, logger) -> dict:
    """Generate the four named variants from the base scene state under run_dir/base/.

    Each variant is written to its own subfolder <run_dir>/<variant_name>/ with
    the canonical hsm_scene_state.json (or stk_scene_state.json) so it can be
    rendered independently via render.py --scene-dir <that subfolder>.
    """
    import scene_variants as var

    base_dir = run_dir / "base"
    hsm_path = base_dir / "hsm_scene_state.json"
    stk_path = base_dir / "stk_scene_state.json"
    if not hsm_path.exists() and not stk_path.exists():
        logger.warning(
            "No scene-state file found in %s; skipping variants. "
            "(Generation may not have placed any objects.)",
            base_dir,
        )
        return {}
    written = var.generate_named_variants(run_dir, seed=seed, source_subdir="base")
    for name, path in written.items():
        logger.info(f"Variant {name} -> {path}")
    return written


# ---------------------------------------------------------------------------
# Rendering (in-process — bpy lives in this same `hsm` env)
# ---------------------------------------------------------------------------

# Rendering runs in-process under the `hsm` conda env, which now carries bpy
# (install via the `[render]` extra: `pip install -e ".[render]"`). bpy + torch
# coexist in one env, so there is no separate render interpreter and no
# subprocess: cli.py imports `render` lazily and calls render_configs() directly.
# bpy itself is only imported when rendering actually runs (the generate-only
# path never touches it).

STATE_FILENAMES = ("hsm_scene_state.json", "stk_scene_state.json")


def _discover_render_subdirs(run_dir: Path) -> list[Path]:
    """Return the scene subdirs to render: base/ (if present) plus every
    variant_*/ subfolder that holds a canonical scene-state file. Sorted for
    deterministic ordering."""
    found: list[Path] = []
    base_dir = run_dir / "base"
    if base_dir.is_dir() and any((base_dir / n).exists() for n in STATE_FILENAMES):
        found.append(base_dir)
    if run_dir.is_dir():
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir() or not child.name.startswith("variant_"):
                continue
            if any((child / n).exists() for n in STATE_FILENAMES):
                found.append(child)
    return found


def _render_configs_for_mode(mode: str) -> list:
    """Build the deduped list of render configs for the given mode (single|all)."""
    import render_config as rcfg

    if mode == "all":
        return rcfg.render_all_configs()
    return [rcfg.single_render_config()]


def _render_subfolders(run_dir: Path, mode: str, hssd_dir: str | None, logger) -> int:
    """Render base/ + each variant_* subfolder in-process (bpy lives in the hsm
    env). Returns the number of subfolders rendered successfully."""
    import render as renderer

    subdirs = _discover_render_subdirs(run_dir)
    if not subdirs:
        logger.warning("No renderable scene subfolders found under %s", run_dir)
        print(
            f"(render: no base/ or variant_*/ with a scene-state file under {run_dir})"
        )
        return 0

    configs = _render_configs_for_mode(mode)
    rendered = 0
    for sub in subdirs:
        logger.info(f"Rendering {sub.name} (mode={mode}): {len(configs)} config(s)")
        print(f"  render [{mode}] {sub.name} ...")
        try:
            renderer.render_configs(sub, configs, hssd_dir)
            rendered += 1
        except Exception as e:
            logger.error(f"Render failed for {sub.name}: {e}")
            logger.error(traceback.format_exc())
            print(f"  render FAILED for {sub.name} ({e}); continuing.")
    return rendered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cli",
        description="Generate an HSM scene from a text prompt, with optional content variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Textual scene description (the LLM prompt). Exactly one of "
        "--prompt / --path must be given.",
    )
    p.add_argument(
        "--path",
        default=None,
        help="Path to an EXISTING run dir (outputs/<datetime>/) to render — "
        "no generation. Exactly one of --prompt / --path must be given.",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for the LLM (env: OPENAI_BASE_URL). "
        "Default official: https://api.openai.com/v1",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM (env: OPENAI_API_KEY, also read from .env).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="LLM model name, e.g. gpt-4o-2024-08-06 (env: OPENAI_MODEL). "
        "Default: gpt-5.1-2025-11-13.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature (env: OPENAI_TEMPERATURE). Default 0.7.",
    )
    p.add_argument(
        "--hssd-dir",
        default=None,
        help="HSSD 3D-models root dir (contains objects/, support-surfaces/). "
        "Default: data/hssd-models. ~72GB dataset — see top docstring.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="HSM data root dir (contains preprocessed/, motif_library/). "
        "Default: data. See top docstring.",
    )
    p.add_argument(
        "--output",
        default="outputs",
        help="Root output folder (a dated sub-folder is created inside it).",
    )
    p.add_argument(
        "--variants",
        action="store_true",
        help="Also generate the four content variants (half / biggest-only / "
        "scrambled / worst-object) from the base scene.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the deterministic variant generators (default 42).",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Render the scene(s) with Blender at the baseline config (512px, "
        "white bg, 50mm, pitch 0 / top-down, yaw 0, city env). With --prompt, "
        "renders the generated base (+ variants if --variants); with --path, "
        "renders base/ + every variant_* already present. Mutually exclusive "
        "with --render-all.",
    )
    p.add_argument(
        "--render-all",
        action="store_true",
        help="Render the full six-factor sweep (resolution / focal / pitch / yaw / "
        "env / background) for each scene subfolder. Same source rules as "
        "--render. Mutually exclusive with --render.",
    )
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    """Enforce mutual exclusions: exactly one of --prompt/--path; not both render flags."""
    if (args.prompt is None) == (args.path is None):
        raise SystemExit(
            "error: pass exactly one of --prompt (generate) or --path (render existing)."
        )
    if args.render and args.render_all:
        raise SystemExit(
            "error: --render and --render-all are mutually exclusive; pick one."
        )
    if args.path and args.variants:
        print(
            "(note: --variants is a generation flag; ignored in --path (render-only) mode)"
        )


def main(argv=None) -> int:
    args = _parse_args(argv)
    _validate_args(args)

    logger = get_logger_safe()
    render_mode = "all" if args.render_all else ("single" if args.render else None)

    # ----- --path: render-only (no generation) ---------------------------------
    if args.path is not None:
        run_dir = Path(args.path).resolve()
        if not run_dir.is_dir():
            print(f"render: --path {run_dir} is not a directory")
            return 1
        print("HSM CLI — render-only mode (no generation)")
        print(f"  run dir:  {run_dir}")
        print(f"  mode:     {render_mode}")
        if render_mode is None:
            print("(--path given without --render/--render-all; nothing to do.)")
            return 0
        n = _render_subfolders(run_dir, render_mode, args.hssd_dir, logger)
        print(f"Rendered {n} scene subfolder(s) under {run_dir}")
        return 0

    # ----- --prompt: generation (+ optional variants + optional render) -------
    # Apply data-path overrides BEFORE importing the pipeline (which transitively
    # imports the retrieval/support modules that read these paths).
    hssd_path, data_path = _apply_data_paths(args.hssd_dir, args.data_dir)
    _apply_llm_env(args)

    print("HSM CLI — scene generation started")
    print(f"Prompt:    {args.prompt}")
    print(
        f"Model:     {args.model or os.environ.get('OPENAI_MODEL') or '(default gpt-5.1)'}"
    )
    print(f"HSSD dir:  {hssd_path}  (exists: {hssd_path.exists()})")
    print(f"Data dir:  {data_path}  (exists: {data_path.exists()})")

    # outputs/<YYYYMMDD-HHMMSS>/  using datetime.datetime.now(datetime.UTC) per spec.
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output).resolve()
    run_dir = out_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(args)
    _write_config_json(run_dir, args, cfg, hssd_path, data_path)

    try:
        context = asyncio.run(_run_pipeline(cfg, run_dir, logger))
    except Exception as e:
        logger.error(f"Scene generation failed: {e}")
        logger.error(traceback.format_exc())
        print(f"HSM Scene generation failed at {stamp} — see {run_dir}/base/scene.log")
        return 1

    written_variants: dict = {}
    if args.variants:
        try:
            written_variants = _generate_variants(run_dir, args.seed, logger)
        except Exception as e:
            logger.error(f"Variant generation failed: {e}")
            logger.error(traceback.format_exc())
            print(
                f"Variant generation failed at {stamp} — base scene still saved in {run_dir}"
            )
            # Don't fail the whole run: the base scene succeeded.
    else:
        print(
            "(skip variants — pass --variants to also generate the four content variants)"
        )

    n_objs = 0
    try:
        scene = context.get("scene")
        if scene is not None:
            n_objs = len(scene.get_all_objects())
    except Exception:
        pass
    print(f"HSM Scene generation succeeded at {stamp}!")
    print(f"  run dir:   {run_dir}")
    print(f"  objects:   {n_objs}")
    if written_variants:
        print(f"  variants:  {len(written_variants)}")

    if render_mode is not None:
        print(f"Rendering scene(s) (mode={render_mode}) ...")
        try:
            n = _render_subfolders(run_dir, render_mode, args.hssd_dir, logger)
            print(f"  rendered:  {n} scene subfolder(s)")
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            logger.error(traceback.format_exc())
            print(f"  rendering failed — scenes are still saved under {run_dir}")
    return 0


def get_logger_safe():
    """Get a logger without forcing heavy imports at module load."""
    from hsm_core.utils import get_logger

    return get_logger("hsm_cli")


if __name__ == "__main__":
    sys.exit(main())
