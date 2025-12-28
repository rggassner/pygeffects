#!venv/bin/python3
import torch
import random
import cv2
import numpy as np
from pathlib import Path
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

# MODEL_ID
# Hugging Face model identifier to load (SDXL base, img2img compatible)
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# MODEL_CACHE
# Local directory where Hugging Face models are stored/cached
MODEL_CACHE = "hf_models"

# PROMPT
# Base positive prompt describing the desired visual content and style
PROMPT = (
    "fractal art of an octopus in the style of alcohol ink, mosaic, "
    "intricate details, fine textures, complex patterns"
)

# NEGATIVE_BASE
# Core negative prompt to suppress common SD issues (artifacts, anatomy, text, etc.)
NEGATIVE_BASE = (
    "blurry, low quality, distorted, low resolution, extra limbs, "
    "mutated hands, artifacts, watermark, text, nsfw"
)

# NEGATIVE_SATURATION
# Extra negative prompt specifically targeting neon, glow, and oversaturation
NEGATIVE_SATURATION = (
    "neon, oversaturated, glowing colors, high contrast gradients"
)

# WIDTH / HEIGHT
# Output resolution for generated images (SDXL performs best at 1024x1024)
WIDTH = HEIGHT = 1024

# STEPS
# Number of diffusion steps per frame (higher = more detail, slower)
STEPS = 35

# CFG
# Classifier-Free Guidance scale (higher = closer to prompt, too high can cause artifacts)
CFG = 7.5

# CFG_RESET_BOOST
# Temporary CFG increase applied during semantic reset frames to re-anchor structure
CFG_RESET_BOOST = 1.0

# BASE_DENOISE
# Starting img2img denoise strength (controls how much the image can change per frame)
BASE_DENOISE = 0.6

# MIN_DENOISE
# Lower bound for adaptive denoise decay (prevents the image from freezing)
MIN_DENOISE = 0.25

# DENOISE_DECAY
# Amount subtracted from denoise per frame to slowly stabilize the image over time
DENOISE_DECAY = 0.0003

# RESET_INTERVAL
# Number of frames between semantic resets (reinjects structure and detail)
RESET_INTERVAL = 20

# RESET_STRENGTH
# Denoise strength used during semantic reset frames
RESET_STRENGTH = 0.75

# MICRO_NOISE_EVERY
# Frequency (in frames) at which small noise is injected to prevent texture collapse
MICRO_NOISE_EVERY = 20

# MICRO_NOISE_AMOUNT
# Blend strength of the injected micro-noise (very small values recommended)
MICRO_NOISE_AMOUNT = 0.015

# SEED
# Random seed for reproducibility (same seed = same evolution, given same parameters)
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
        "BASE_DENOISE": 0.55,
        "DENOISE_DECAY": 0.00025,
        "MICRO_NOISE_AMOUNT": 0.02,
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

# =========================================================
# SEGMENT VALUE RESOLVER
# =========================================================

def segval(seg, key, default):
    return seg.get(key, default)

# =========================================================
# IMAGE TRANSFORM
# =========================================================

def apply_transform(img, zoom=1.0, rotate=0.0, shift_x=0, shift_y=0):
    w, h = img.size

    if zoom != 1.0:
        nw, nh = int(w * zoom), int(h * zoom)
        img = img.resize((nw, nh), Image.LANCZOS)
        img = img.crop((
            (nw - w) // 2,
            (nh - h) // 2,
            (nw + w) // 2,
            (nh + h) // 2
        ))

    if rotate != 0.0:
        img = img.rotate(rotate, resample=Image.BICUBIC, expand=False)

    if shift_x or shift_y:
        shifted = Image.new("RGB", (w, h))
        shifted.paste(img, (shift_x, shift_y))
        img = shifted

    return img

# =========================================================
# MICRO NOISE
# =========================================================

def add_micro_noise(img, amount):
    noise = Image.effect_noise(img.size, random.uniform(2, 8)).convert("RGB")
    return Image.blend(img, noise, amount)

# =========================================================
# BLACK INPAINT
# =========================================================

def remove_black_pixels(img_pil):
    img_rgb = np.array(img_pil)

    lower = np.clip(
        np.array(BLACK_TARGET_COLOR) - BLACK_COLOR_RANGE,
        0, 255
    ).astype(np.uint8)

    upper = np.clip(
        np.array(BLACK_TARGET_COLOR) + BLACK_COLOR_RANGE,
        0, 255
    ).astype(np.uint8)

    mask = cv2.inRange(img_rgb, lower, upper)

    if BLACK_MASK_BLUR > 0:
        mask = cv2.GaussianBlur(mask, (BLACK_MASK_BLUR, BLACK_MASK_BLUR), 0)

    inpainted = cv2.inpaint(
        img_rgb,
        mask,
        BLACK_INPAINT_RADIUS,
        cv2.INPAINT_TELEA
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
generator = torch.Generator(device="cuda").manual_seed(SEED)

# =========================================================
# MAIN LOOP
# =========================================================

img = Image.open("input.png").convert("RGB").resize((WIDTH, HEIGHT))
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
            generator=generator,
        )

        img = result.images[0]

        out_path = out_dir / f"frame_{frame_idx:05d}.png"
        img.save(out_path)

        print(
            f"Saved {out_path} | "
            f"denoise={strength:.3f} | cfg={guidance:.2f}"
        )

        frame_idx += 1

