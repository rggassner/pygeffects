#!venv/bin/python3
"""
Minimal SDXL text-to-image generator with CLI parameters.

Generates one or more images from a text prompt using
Stable Diffusion XL (text-to-image mode).

Designed as a clean starting point for experimentation
and extension.
"""

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline

MODEL_CACHE = "hf_models"


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple SDXL text-to-image generator"
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Positive prompt describing the image"
    )

    parser.add_argument(
        "--negative",
        default="(worst quality, low quality:1.4), watermark, text",
        help="Negative prompt"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="Number of diffusion steps (default: 35)"
    )

    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="CFG scale / guidance strength (default: 7.5)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random)"
    )

    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)"
    )

    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory (default: ./outputs)"
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir=MODEL_CACHE,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")

    pipe.set_progress_bar_config(disable=False)

    generator = (
        torch.Generator(device="cuda").manual_seed(args.seed)
        if args.seed is not None
        else None
    )

    for i in range(args.num):
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
            generator=generator,
        )

        img = result.images[0]

        seed_suffix = args.seed if args.seed is not None else "random"
        out_path = out_dir / f"img_{i:03d}_seed_{seed_suffix}.png"
        img.save(out_path)

        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
