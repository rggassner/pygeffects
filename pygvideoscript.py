#!venv/bin/python3
import os
import sys
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
import numpy as np
import cv2

# =========================
# Defaults
# =========================

WIDTH = 1024
HEIGHT = 1024
STEPS = 35
GUIDANCE_SCALE = 7.5
SEED = 1337970693
DENOISING_STRENGTH = 0.6
TEMPORAL_STRENGTH = 0.6  # how much previous frame dominates

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE = "hf_models"

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


def merge_temporal(prev_bgr, curr_bgr, strength):
    """
    strength = how much previous generated frame dominates
    """
    assert 0.0 <= strength <= 1.0

    prev_lab = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2LAB)
    curr_lab = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2LAB)

    merged = curr_lab.copy()

    # Blend luminance
    merged[:, :, 0] = (
        prev_lab[:, :, 0] * strength +
        curr_lab[:, :, 0] * (1.0 - strength)
    ).astype(np.uint8)

    # Color mostly from previous frame to reduce flicker
    merged[:, :, 1:] = (
        prev_lab[:, :, 1:] * 0.8 +
        curr_lab[:, :, 1:] * 0.2
    ).astype(np.uint8)

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

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

def main(): #pylint: disable=too-many-statements, too-many-locals
    """
    Main entry point for the video-to-SDXL img2img processing pipeline.

    This function parses command-line arguments, loads a Stable Diffusion XL
    img2img model, iterates over frames of an input video, optionally applies
    color-based masking and replacement, performs temporal blending with the
    previous generated frame, and finally runs SDXL img2img to generate a
    stylized output frame for each video frame.

    High-level pipeline:
    1. Parse CLI arguments that control video input, output paths, masking
       behavior, SDXL parameters, and temporal blending strength.
    2. Load the SDXL img2img pipeline with GPU acceleration.
    3. Open the input video and iterate frame by frame.
    4. Resize frames to the target resolution.
    5. Detect pixels within a target color range and optionally replace them
       using Gaussian noise or inpainting.
    6. Merge the current frame with the previously generated frame to enforce
       temporal coherence.
    7. Run SDXL img2img using the processed frame as the init image.
    8. Save each generated frame with a zero-padded sequential filename.

    The output is a directory of generated PNG frames suitable for later
    post-processing, such as frame interpolation or video reassembly.

    Command-line arguments control:
    - Masking method ("none", "gaussian", "inpaint")
    - Target color and tolerance for masking
    - Noise or inpainting parameters
    - SDXL prompt, negative prompt, and generation settings
    - Temporal blending strength between consecutive frames

    This function exits the program if the input video cannot be opened.
    """
    parser = argparse.ArgumentParser(
        description="Video, temporal merge, SDXL img2img"
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

    parser.add_argument(
        "--temporal-strength",
        type=float,
        default=TEMPORAL_STRENGTH,
        help="How much previous generated frame influences the next (0-1)"
    )

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

    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # =========================
    # Video processing
    # =========================

    cap = cv2.VideoCapture(args.video_file) # pylint: disable=no-member
    if not cap.isOpened():
        print("Could not open video")
        sys.exit(1)

    frame_index = 0
    prev_generated_bgr = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize( # pylint: disable=no-member
            frame,
            (args.width, args.height),
            interpolation=cv2.INTER_AREA # pylint: disable=no-member
        )

        image_bgr = frame
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # pylint: disable=no-member

        mask = color_range_mask(
            image_rgb,
            tuple(args.target_color),
            args.color_range
        )

        if args.method == "gaussian" and mask.any():
            image_bgr = gaussian_replace(image_bgr, mask, args.noise_level)

        elif args.method == "inpaint" and mask.any():
            image_rgb = inpaint_replace(image_rgb, mask, args.inpaint_radius)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # pylint: disable=no-member

        # =========================
        # Temporal merge
        # =========================

        if prev_generated_bgr is not None:
            image_bgr = merge_temporal(
                prev_generated_bgr,
                image_bgr,
                args.temporal_strength
            )

        # =========================
        # SDXL img2img
        # =========================

        init_image = Image.fromarray(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # pylint: disable=no-member
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
        output_path = os.path.join(args.output_path, filename)
        result.save(output_path)

        prev_generated_bgr = cv2.cvtColor( # pylint: disable=no-member
            np.array(result),
            cv2.COLOR_RGB2BGR # pylint: disable=no-member
        )

        print(f"Frame {frame_index} done")
        frame_index += 1

    cap.release()
    print("Done")

if __name__ == "__main__":
    main()
