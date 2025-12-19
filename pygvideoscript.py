#!venv/bin/python3
import cv2
import numpy as np
import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

# =========================
# Defaults
# =========================

WIDTH = 1024
HEIGHT = 1024
STEPS = 35
GUIDANCE_SCALE = 7.5
SEED = 1337970693
DENOISING_STRENGTH = 0.45

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE = "/home/rgg/hf_models"

# =========================
# Helpers
# =========================

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def color_range_mask(image_rgb, target_color, color_range):
    target = np.array(target_color, dtype=np.int16)
    lower = np.clip(target - color_range, 0, 255).astype(np.uint8)
    upper = np.clip(target + color_range, 0, 255).astype(np.uint8)
    return cv2.inRange(image_rgb, lower, upper)

# =========================
# Noise / Inpainting
# =========================

def gaussian_replace(image, mask, noise_level):
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    img = image.astype(np.int16)
    img[mask > 0] += noise[mask > 0]
    return np.clip(img, 0, 255).astype(np.uint8)


def inpaint_replace(image_rgb, mask, radius):
    return cv2.inpaint(
        image_rgb,
        mask,
        inpaintRadius=radius,
        flags=cv2.INPAINT_TELEA
    )

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Video → frames → optional noise → SDXL img2img"
    )

    parser.add_argument("video_file")
    parser.add_argument("--output-path", default="video_input_frames")

    parser.add_argument("--method",
        choices=["none", "gaussian", "inpaint"],
        default="none"
    )

    parser.add_argument("--target-color", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--color-range", type=int, default=30)
    parser.add_argument("--noise-level", type=int, default=50)
    parser.add_argument("--inpaint-radius", type=int, default=30)

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")

    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--denoising-strength", type=float, default=DENOISING_STRENGTH)

    args = parser.parse_args()

    ensure_output_dir(args.output_path)

    # =========================
    # Load SDXL
    # =========================

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")

#    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_xformers_memory_efficient_attention = False

    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # =========================
    # Video processing
    # =========================

    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        print("Could not open video")
        sys.exit(1)

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(
            frame,
            (args.width, args.height),
            interpolation=cv2.INTER_AREA
        )

        image_bgr = frame
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = color_range_mask(
            image_rgb,
            tuple(args.target_color),
            args.color_range
        )

        if args.method == "gaussian" and mask.any():
            image_bgr = gaussian_replace(image_bgr, mask, args.noise_level)

        elif args.method == "inpaint" and mask.any():
            image_rgb = inpaint_replace(image_rgb, mask, args.inpaint_radius)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # =========================
        # SDXL img2img
        # =========================

        init_image = Image.fromarray(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        )

        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=init_image,
            strength=args.denoising_strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        filename = f"{frame_index:016d}.png"
        result.save(os.path.join(args.output_path, filename))

        print(f"Frame {frame_index} done")
        frame_index += 1

    cap.release()
    print("Done")

if __name__ == "__main__":
    main()

