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

MODEL_CACHE = "/home/rgg/hf_models"


# =========================================================
# CLI
# =========================================================

def parse_args():
    """
    Parse and validate command-line arguments for the SDXL text-to-image generator.

    This function defines all supported CLI options used to control image
    generation parameters such as prompts, diffusion settings, image
    resolution, reproducibility, and output handling. It returns an
    argparse.Namespace object containing the parsed values.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the following
        attributes:
            prompt (str): Positive text prompt describing the desired image.
            negative (str): Negative prompt to suppress unwanted features.
            steps (int): Number of diffusion steps to run.
            cfg (float): Classifier-free guidance scale controlling prompt strength.
            width (int): Width of the generated image in pixels.
            height (int): Height of the generated image in pixels.
            seed (int or None): Random seed for reproducibility, or None for random.
            num (int): Number of images to generate.
            out (str): Output directory where generated images will be saved.
    """
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
    """
    Entry point for batch image generation using Stable Diffusion XL (text-to-image).

    This function parses command-line arguments, initializes a Stable Diffusion XL
    pipeline, and generates one or more images from a text prompt. Each image is
    produced independently, optionally using a fixed random seed for reproducible
    results, and saved to the specified output directory with an indexed filename.

    The pipeline runs on CUDA using half-precision (float16) tensors and relies on
    locally cached model files. Progress reporting is enabled for visibility during
    generation.

    Workflow
    --------
    1. Parse command-line arguments (prompt, output directory, generation settings).
    2. Create the output directory if it does not already exist.
    3. Load the Stable Diffusion XL base model and move it to the GPU.
    4. Initialize a CUDA random number generator if a seed is provided.
    5. Generate the requested number of images using the same prompt and settings.
    6. Save each generated image to disk with an index and seed identifier.

    Notes
    -----
    - If a seed is provided, all images are generated deterministically from that
      seed; otherwise, each image uses a random seed.
    - Images are generated directly from text (text-to-image), not img2img.
    - The output filenames include the seed value (or 'random') for traceability.

    Side Effects
    ------------
    - Writes PNG image files to the output directory.
    - Prints the path of each saved image to stdout.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded, CUDA is unavailable, or inference fails.
    """
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
