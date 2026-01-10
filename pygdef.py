#!venv/bin/python3
"""
Grid-based image outpainting using Stable Diffusion inpainting.

This script scrapes a remote grid-style image layout, reconstructs the
available tiles into a composite canvas, and uses Stable Diffusion
inpainting to synthesize missing regions. The generated result is then
iteratively refined to produce multiple variations, which are saved
both as full images and as individual grid tiles.

Core features
-------------
- Scrapes a 3×3 grid of images from a target website.
- Rebuilds the known tiles into a base image.
- Automatically constructs an inpainting mask for missing cells.
- Uses a cached Stable Diffusion inpainting model for fast iteration.
- Supports optional image embedding at user-defined coordinates.
- Generates multiple output variations in a single run.
- Saves per-iteration outputs in timestamped directories.

Intended use
------------
This tool is designed for experimental image exploration, procedural
outpainting, and grid-based visual expansion workflows. It is especially
useful when working with tiled or partially-known image layouts that
benefit from generative completion.

Requirements
------------
- CUDA-capable GPU
- PyTorch
- diffusers
- Pillow (PIL)
- BeautifulSoup4
- requests

Execution
---------
Run the script from the command line and provide prompts and parameters
via CLI arguments. See ``--help`` for details.

Note
----
NSFW filtering can be disabled via configuration. Use responsibly and
ensure compliance with model and content policies.
"""
from pathlib import Path
import random
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

# =========================================================
# USER-TUNABLE BEHAVIOR FLAGS
# =========================================================

USE_DPM = True
USE_KARRAS = True
USE_ADAPTIVE_DENOISE = True
USE_SEMANTIC_RESET = True
USE_SATURATION_SUPPRESSION = True
USE_MICRO_NOISE = True
USE_BLACK_INPAINT = True

# =========================================================
# 🎛 GLOBAL DEFAULT PARAMETERS
# =========================================================

# MODEL_ID
# Hugging Face model identifier to load (SDXL base, img2img compatible)
# default: "stabilityai/stable-diffusion-xl-base-1.0"
# range: SDXL-compatible models only
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# MODEL_CACHE
# Local directory where Hugging Face models are cached
# default: "/home/rgg/hf_models"
# range: any valid local path with read access
MODEL_CACHE = "hf_models"


# PROMPT
# Base positive prompt defining subject, style, and visual intent
# default: fractal octopus alcohol ink mosaic
# range: free text; keep concise but descriptive for recursion
PROMPT = (
        "A father and son statue in Madame Tussauds wax museum "
)


# NEGATIVE_BASE
# Core negative prompt suppressing artifacts, anatomy issues, text, etc.
# default: generic SDXL cleanup negatives
# range: add/remove terms carefully; too long may weaken guidance
NEGATIVE_BASE = (
        " (worst quality, low quality:1.4), (watermark), censored, two katana,"
)


# NEGATIVE_SATURATION
# Extra negative prompt to suppress neon, glow, and oversaturation drift
# default: neon suppression
# range: useful for long img2img runs; can be disabled if desired
NEGATIVE_SATURATION = (
    "neon, oversaturated, glowing colors, high contrast gradients"
)


# WIDTH / HEIGHT
# Output resolution of generated frames
# default: 1024
# range: 1024 recommended for SDXL; avoid non-square unless intentional
WIDTH = HEIGHT = 1024


# STEPS
# Number of diffusion steps per frame
# default: 35
# range: 20–50 (higher = more detail, slower, diminishing returns >40)
STEPS = 35


# CFG
# Classifier-Free Guidance scale (prompt adherence strength)
# default: 7.5
# range: 5.5–9.0 (too high may cause artifacts or color burn-in)
CFG = 6.5


# CFG_RESET_BOOST
# Extra CFG added during semantic reset frames
# default: 1.5
# range: 0.5–3.0 (higher = stronger structure reassertion)
CFG_RESET_BOOST = .5


# BASE_DENOISE
# Initial img2img denoise strength (controls per-frame evolution)
# default: 0.6
# range: 0.45–0.7 (too high = chaos, too low = stagnation)
BASE_DENOISE = 0.45


# MIN_DENOISE
# Lower bound for adaptive denoise decay
# default: 0.25
# range: 0.2–0.35 (prevents the image from freezing)
MIN_DENOISE = 0.25


# DENOISE_DECAY
# Amount subtracted from denoise per frame
# default: 0.0003
# range: 0.0001–0.001 (higher = faster stabilization)
DENOISE_DECAY = 0.0003


# RESET_INTERVAL
# Number of frames between semantic resets
# default: 50
# range: 20–150 (shorter = more structure, longer = more drift)
RESET_INTERVAL = 20


# RESET_STRENGTH
# Denoise strength used during semantic reset frames
# default: 0.75
# range: 0.6–0.85 (too high can overwrite evolution)
RESET_STRENGTH = 0.6


# MICRO_NOISE_EVERY
# Frequency (in frames) to inject subtle noise
# default: 20
# range: 10–50 (lower = more texture refresh, higher = calmer evolution)
MICRO_NOISE_EVERY = 20


# MICRO_NOISE_AMOUNT
# Blend strength of injected micro-noise
# default: 0.015
# range: 0.005–0.03 (keep small to avoid grain buildup)
MICRO_NOISE_AMOUNT = 0.015


# SEED
# Random seed for reproducibility
# default: 123456
# range: any integer; fixed = deterministic evolution
SEED = 123456


# =========================================================
# BLACK INPAINT PARAMETERS
# =========================================================

BLACK_TARGET_COLOR = (0, 0, 0)
BLACK_COLOR_RANGE = 8
BLACK_INPAINT_RADIUS = 3
BLACK_MASK_BLUR = 3

# =========================================================
# SEGMENTS (NOW SUPPORT OVERRIDES)
# =========================================================

SEGMENTS = [
    {
        "frames": 1000,
        "zoom_per_frame": 1.01,
        "rotate_per_frame": -0.1,
        "shift_x_per_frame": 1,
        "shift_y_per_frame": 0,

        # Optional overrides
        #"BASE_DENOISE": 0.55,
        #"DENOISE_DECAY": 0.00025,
        #"MICRO_NOISE_AMOUNT": 0.02,
    },
    {
        "frames": 100,
        "zoom_per_frame": 1.0,
        "rotate_per_frame": 1.1,
        "shift_x_per_frame": 0,
        "shift_y_per_frame": 2,

        "PROMPT": "highly intricate abstract fractal ink illustration",
        "CFG": 8.5,
        "RESET_INTERVAL": 25,
    },
]


def segval(segment, key, default):
    """
    Retrieves a configuration value from a segment dictionary with fallback.

    This helper function allows per-segment parameter overrides while
    gracefully falling back to a global or default value when the key
    is not defined in the segment.

    It is used to unify access to tunable parameters across animation
    segments without scattering conditional logic throughout the
    main generation loop.

    Parameters
    ----------
    segment : dict
        Segment configuration dictionary that may contain overrides.
    key : str
        Name of the parameter to retrieve.
    default : Any
        Value to return if the key is not present in the segment.

    Returns
    -------
    Any
        The segment-specific value if defined, otherwise the default.
    """
    return segment.get(key, default)


def apply_transform(image, zoom=1.0, rotate=0.0, shift_x=0, shift_y=0):
    """
    Applies deterministic geometric transformations to an image while
    preserving the original output dimensions.

    The transformation sequence consists of:
    1) Centered zoom with resize and crop to maintain resolution
    2) In-place rotation around the image center
    3) Pixel translation via x/y shifts with empty regions filled in black

    This function is designed for frame-to-frame evolution in generative
    pipelines, where small, controlled spatial changes accumulate over time
    without altering image size or aspect ratio.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image to be transformed.
    zoom : float, optional
        Scale factor applied uniformly to width and height.
        Values > 1.0 zoom in, values < 1.0 zoom out.
    rotate : float, optional
        Rotation angle in degrees. Positive values rotate counter-clockwise.
    shift_x : int, optional
        Horizontal pixel shift. Positive values move the image right.
    shift_y : int, optional
        Vertical pixel shift. Positive values move the image down.

    Returns
    -------
    PIL.Image.Image
        Transformed image with the same dimensions as the input.

    Notes
    -----
    - Zooming is center-cropped to avoid resolution drift.
    - Rotation does not expand the canvas; corners may be clipped.
    - Shifting introduces black padding, which can be handled downstream
      by inpainting or diffusion-based correction.
    """
    w, h = image.size

    if zoom != 1.0:
        nw, nh = int(w * zoom), int(h * zoom)
        image = image.resize((nw, nh), Image.LANCZOS) # pylint: disable=no-member
        image = image.crop((
            (nw - w) // 2,
            (nh - h) // 2,
            (nw + w) // 2,
            (nh + h) // 2
        ))

    if rotate != 0.0:
        image = image.rotate(rotate, resample=Image.BICUBIC, expand=False) # pylint: disable=no-member

    if shift_x or shift_y:
        shifted = Image.new("RGB", (w, h))
        shifted.paste(image, (shift_x, shift_y))
        image = shifted

    return image


def add_micro_noise(image, amount):
    """
    Injects subtle, low-frequency noise into an image to refresh texture
    and prevent visual stagnation during recursive img2img evolution.

    A procedurally generated noise layer is blended with the input image
    using a small alpha value, reintroducing micro-variation without
    visibly degrading structure or coherence.

    This technique helps counteract over-smoothing, color banding, and
    detail collapse across long generation runs.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image to receive micro-noise injection.
    amount : float
        Blend strength of the noise layer.
        Typical range: 0.005–0.03.

    Returns
    -------
    PIL.Image.Image
        Image with subtle noise blended in.
    """
    noise = Image.effect_noise(image.size, random.uniform(2, 8)).convert("RGB")
    return Image.blend(image, noise, amount)


def remove_black_pixels(img_pil):
    """
    Detects near-black pixels in an image and removes them via OpenCV inpainting.

    This function identifies pixels close to a target black color using a
    configurable tolerance range, builds a binary mask of those regions,
    optionally blurs the mask to soften edges, and then applies Telea
    inpainting to reconstruct the affected areas from surrounding context.

    It is primarily intended to clean up black seams, voids, or artifacts
    introduced by geometric transforms (zoom, rotation, shifting) during
    recursive img2img pipelines.

    Parameters
    ----------
    img_pil : PIL.Image.Image
        Input RGB image to be processed.

    Returns
    -------
    PIL.Image.Image
        Image with black or near-black regions inpainted and visually blended
        with surrounding pixels.

    Notes
    -----
    - The target color, tolerance range, blur size, and inpaint radius are
      controlled by the global BLACK_* configuration variables.
    - OpenCV functions are dynamically bound (C-extension); pylint warnings
      for `cv2` members are safely suppressed.
    - Inpainting uses the Telea algorithm, which favors smooth and natural
      reconstruction suitable for generative art workflows.
    """
    img_rgb = np.array(img_pil)

    lower = np.clip(
        np.array(BLACK_TARGET_COLOR) - BLACK_COLOR_RANGE,
        0, 255
    ).astype(np.uint8)

    upper = np.clip(
        np.array(BLACK_TARGET_COLOR) + BLACK_COLOR_RANGE,
        0, 255
    ).astype(np.uint8)

    mask = cv2.inRange(img_rgb, lower, upper) # pylint: disable=no-member

    if BLACK_MASK_BLUR > 0:
        mask = cv2.GaussianBlur(mask, (BLACK_MASK_BLUR, BLACK_MASK_BLUR), 0) # pylint: disable=no-member

    inpainted = cv2.inpaint( # pylint: disable=no-member
        img_rgb,
        mask,
        BLACK_INPAINT_RADIUS,
        cv2.INPAINT_TELEA # pylint: disable=no-member
    )

    return Image.fromarray(inpainted)

# =========================================================
# OUTPUT
# =========================================================

out_dir = Path("pygdef")
out_dir.mkdir(exist_ok=True)

# =========================================================
# PIPELINE INIT
# =========================================================

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=MODEL_CACHE,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
).to("cuda")

pipe.scheduler = (
    DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        solver_order=2,
        use_karras_sigmas=USE_KARRAS,
    )
    if USE_DPM
    else EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
)

pipe.set_progress_bar_config(disable=True)
GENERATOR = torch.Generator(device="cuda").manual_seed(SEED)

# =========================================================
# MAIN LOOP
# =========================================================

img = Image.open("input1.png").convert("RGB").resize((WIDTH, HEIGHT))
frame_idx = 0

for seg in SEGMENTS:
    for _ in range(seg["frames"]):

        base_denoise = segval(seg, "BASE_DENOISE", BASE_DENOISE)
        min_denoise = segval(seg, "MIN_DENOISE", MIN_DENOISE)
        decay = segval(seg, "DENOISE_DECAY", DENOISE_DECAY)

        denoise = (
            max(min_denoise, base_denoise - frame_idx * decay)
            if USE_ADAPTIVE_DENOISE
            else base_denoise
        )

        img = apply_transform(
            img,
            zoom=seg["zoom_per_frame"],
            rotate=seg["rotate_per_frame"],
            shift_x=seg["shift_x_per_frame"],
            shift_y=seg["shift_y_per_frame"],
        )

        if USE_BLACK_INPAINT:
            img = remove_black_pixels(img)

        if USE_MICRO_NOISE and frame_idx % segval(seg, "MICRO_NOISE_EVERY", MICRO_NOISE_EVERY) == 0:
            img = add_micro_noise(
                img,
                segval(seg, "MICRO_NOISE_AMOUNT", MICRO_NOISE_AMOUNT),
            )

        prompt = segval(seg, "PROMPT", PROMPT)
        negative = segval(seg, "NEGATIVE_BASE", NEGATIVE_BASE)

        if USE_SATURATION_SUPPRESSION:
            negative += ", " + segval(seg, "NEGATIVE_SATURATION", NEGATIVE_SATURATION)

        reset_interval = segval(seg, "RESET_INTERVAL", RESET_INTERVAL)

        if USE_SEMANTIC_RESET and frame_idx > 0 and frame_idx % reset_interval == 0:
            strength = segval(seg, "RESET_STRENGTH", RESET_STRENGTH)
            guidance = segval(seg, "CFG", CFG) + segval(seg, "CFG_RESET_BOOST", CFG_RESET_BOOST)
        else:
            strength = denoise
            guidance = segval(seg, "CFG", CFG)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=img,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=segval(seg, "STEPS", STEPS),
            generator=GENERATOR,
        )

        img = result.images[0]

        out_path = out_dir / f"frame_{frame_idx:05d}.png"
        img.save(out_path)

        print(
            f"Saved {out_path} | "
            f"denoise={strength:.3f} | cfg={guidance:.2f}"
        )

        frame_idx += 1
