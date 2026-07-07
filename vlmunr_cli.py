#!/usr/bin/env python3
"""
vlmunr_cli.py — Single-command scene generation + content-variant driver for the
VLM-unreliability audit harness.

Drives HSM's scene-generation pipeline directly from a textual scene description,
then (optionally) writes the four content variants derived from the base scene —
WITHOUT regenerating the scene or re-calling the LLM:

    variant_01_half          keep round(n/2) objects (seeded sample)
    variant_02_biggest-only  keep the single largest object (by bbox volume)
    variant_03_scrambled     randomize every object's position AND heading in-room
    variant_04_worst-object  fork + re-run retrieval with the CLIP ranking inverted
                             so each object's asset is swapped for the worst match

LLM connection (--base-url/--api-key/--model/--temperature) is wired through the
VLMUNR_OPENAI_* env vars, which hsm_core.vlm.gpt.Session reads at construction.
No source files in the pipeline need parameter threading.

Output layout (per run), under ./outputs/:

    outputs/<YYYYMMDD-HHMMSS>/
        config.json                       prompt + all CLI/LLM configs
        room_scene.glb                    base scene GLB
        hsm_scene_state.json              base scene state (mesh_path + pose)
        stk_scene_state.json              base scene state (column-major STK)
        scene.log                         run log
        visualizations/ ...               stage visualizations
        scene_motifs/ ...                 motif GLBs/pickles
        hsm_scene_state_variant_01_half.json
        hsm_scene_state_variant_02_biggest-only.json
        hsm_scene_state_variant_03_scrambled.json
        hsm_scene_state_variant_04_worst-object.json   (only with --variants)

Usage:
    conda activate hsm
    python vlmunr_cli.py --prompt "A cozy bedroom with a bed and nightstand." \
        --base-url https://api.openai.com/v1 --api-key sk-... \
        --model gpt-4o-2024-08-06 --temperature 0.7 --variants
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

# Import lazily-friendly pipeline pieces. These pull in torch/openai; importing at
# module top keeps the CLI self-contained and surfaces env errors early.
from argparser import HSMArgumentParser
from hsm_core.scene.processing import (
    setup_scene_generation,
    create_processing_pipeline,
    process_cleanup_stage,
)
from hsm_core.utils import get_logger

import vlmunr_variants as var


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> DictConfig | ListConfig:
    """Build the HSM config from the scene_config.yaml + CLI overrides.

    Reuses HSMArgumentParser so the config schema matches `main.py` exactly,
    then layers on the LLM block and result_dir.
    """
    parser = HSMArgumentParser(PROJECT_ROOT)
    # HSMArgumentParser requires -d; fabricate a dummy Namespace to reuse get_config.
    ns = argparse.Namespace(
        desc=args.prompt,
        output="outputs",  # overridden below; the real run dir is set via output_dir_override
        types=["large", "wall", "ceiling", "small"],
        extra_types=["large", "wall", "ceiling", "small"],
        skip_scene_motifs=False,
        skip_solver=False,
        skip_spatial_optimization=False,
    )
    cfg = parser.get_config(ns)

    # Inject an `llm` block so model-type routing picks 'gpt'. The actual model
    # name / temperature reach the Session via VLMUNR_OPENAI_* env vars (set in
    # _apply_llm_env), which every create_session() site reads uniformly.
    llm_block = OmegaConf.create({"model_type": "gpt", "model_name": args.model})
    cfg.llm = llm_block

    cfg.execution.result_dir = str(Path(args.output).resolve())
    return cfg


def _apply_llm_env(args: argparse.Namespace) -> None:
    """Export VLMUNR_OPENAI_* env vars consumed by hsm_core.vlm.gpt.Session."""
    if args.base_url is not None:
        os.environ["VLMUNR_OPENAI_BASE_URL"] = args.base_url
    if args.api_key is not None:
        os.environ["VLMUNR_OPENAI_API_KEY"] = args.api_key
    if args.model is not None:
        os.environ["VLMUNR_OPENAI_MODEL"] = args.model
    if args.temperature is not None:
        os.environ["VLMUNR_OPENAI_TEMPERATURE"] = str(args.temperature)


def _write_config_json(run_dir: Path, args: argparse.Namespace, cfg: DictConfig | ListConfig) -> Path:
    """Persist the prompt + all configs for reproducibility."""
    config_path = run_dir / "config.json"
    payload = {
        "prompt": args.prompt,
        "llm": {
            "base_url": args.base_url or os.environ.get("VLMUNR_OPENAI_BASE_URL"),
            "model": args.model or os.environ.get("VLMUNR_OPENAI_MODEL"),
            "temperature": args.temperature if args.temperature is not None else 0.7,
            "api_key": "<redacted>" if (args.api_key or os.environ.get("VLMUNR_OPENAI_API_KEY")) else None,
        },
        "variants": args.variants,
        "variant_seed": args.seed,
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
    start_time = time.time()
    logger.info("Starting scene generation pipeline (vlmunr_cli)")

    context = await setup_scene_generation(
        cfg, project_root=PROJECT_ROOT,
        output_dir_override=run_dir,   # write everything under outputs/<timestamp>/
        timestamp=False,               # no extra timestamp subdir; run_dir IS the dir
    )
    context["start_time"] = start_time
    pipeline = create_processing_pipeline(cfg, context["is_loaded_scene"])
    logger.info(f"Pipeline created with {len(pipeline)} stages")

    for stage_name, stage_func in pipeline:
        logger.info(f"Stage: {stage_name}")
        context = await stage_func(context, cfg)

    # The pipeline's final stage already saved once (save_scene_state=False default).
    # Re-save WITH save_scene_state=True so hsm_scene_state.json is emitted — this is
    # what the variants layer and the renderer prefer (carries mesh_path + pose,
    # no HSSD_DIR needed). recreate_scene=True reflects any in-memory mutations.
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
    """Generate the four named variants from the base scene state in run_dir."""
    # Prefer hsm_scene_state.json (richer); fall back to stk via _load.
    hsm_path = run_dir / "hsm_scene_state.json"
    stk_path = run_dir / "stk_scene_state.json"
    if not hsm_path.exists() and not stk_path.exists():
        logger.warning(
            "No scene-state file found in %s; skipping variants. "
            "(Generation may not have placed any objects.)", run_dir
        )
        return {}
    written = var.generate_named_variants(run_dir, seed=seed)
    for name, path in written.items():
        logger.info(f"Variant {name} -> {path}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vlmunr_cli",
        description="Generate an HSM scene from a text prompt, with optional content variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-p", "--prompt", required=True,
                   help="Textual scene description (the LLM prompt).")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base URL for the LLM (env: VLMUNR_OPENAI_BASE_URL).")
    p.add_argument("--api-key", default=None,
                   help="API key for the LLM (env: VLMUNR_OPENAI_API_KEY).")
    p.add_argument("--model", default=None,
                   help="LLM model name, e.g. gpt-4o-2024-08-06 (env: VLMUNR_OPENAI_MODEL).")
    p.add_argument("--temperature", type=float, default=None,
                   help="LLM sampling temperature (env: VLMUNR_OPENAI_TEMPERATURE).")
    p.add_argument("--output", default="outputs",
                   help="Root output folder (a dated sub-folder is created inside it).")
    p.add_argument("--variants", action="store_true",
                   help="Also generate the four content variants (half / biggest-only / "
                        "scrambled / worst-object) from the base scene.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the deterministic variant generators (default 42).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    _apply_llm_env(args)

    logger = get_logger("vlmunr_cli")
    print("VLMUNR CLI — scene generation started")
    print(f"Prompt: {args.prompt}")
    print(f"Model:  {args.model or os.environ.get('VLMUNR_OPENAI_MODEL') or '(default gpt-5.1)'}")

    # outputs/<YYYYMMDD-HHMMSS>/  using datetime.datetime.now(datetime.UTC) per spec.
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output).resolve()
    run_dir = out_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(args)
    _write_config_json(run_dir, args, cfg)

    try:
        context = asyncio.run(_run_pipeline(cfg, run_dir, logger))
    except Exception as e:
        logger.error(f"Scene generation failed: {e}")
        logger.error(traceback.format_exc())
        print(f"HSM Scene generation failed at {stamp} — see {run_dir}/scene.log")
        return 1

    written_variants: dict = {}
    if args.variants:
        try:
            written_variants = _generate_variants(run_dir, args.seed, logger)
        except Exception as e:
            logger.error(f"Variant generation failed: {e}")
            logger.error(traceback.format_exc())
            print(f"Variant generation failed at {stamp} — base scene is still saved in {run_dir}")
            # Don't fail the whole run: the base scene succeeded.
    else:
        print("(skip variants — pass --variants to also generate the four content variants)")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
