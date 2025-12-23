# Video to SDXL Img2Img Pipeline with Temporal Consistency

This script processes a video frame by frame, applies optional color-based masking and noise or inpainting, enforces temporal consistency between frames, and then runs each frame through Stable Diffusion XL img2img to generate a new sequence of images suitable for video reconstruction.

The goal is to stylize or transform a video while reducing flicker and preserving temporal coherence between consecutive frames.

---

## Features

* Frame-by-frame video processing using OpenCV
* Optional color-based masking
* Mask replacement strategies:

  * Gaussian noise injection
  * OpenCV inpainting
* Temporal frame merging to reduce flicker
* Stable Diffusion XL img2img inference
* Deterministic output via fixed random seed
* Outputs sequentially numbered PNG frames

---

## Requirements

### System

* Linux recommended
* NVIDIA GPU with CUDA support

### Python

* Python 3.10+

### Python Dependencies

* torch (CUDA-enabled)
* diffusers
* transformers
* accelerate
* opencv-python
* pillow
* numpy

Install dependencies using pip:

```bash
pip install torch diffusers transformers accelerate opencv-python pillow numpy
```

---

## Model Setup

The script uses Stable Diffusion XL Base 1.0 and expects the model to already be available locally.

By default:

* Model ID: `stabilityai/stable-diffusion-xl-base-1.0`
* Cache directory: `/home/rgg/hf_models`

The pipeline is loaded with `local_files_only=True`, so the model must be present in the cache directory.

---

## Usage

```bash
./script.py input_video.mp4 \
  --prompt "your prompt here" \
  --negative-prompt "optional negative prompt" \
  --output-path video_input_frames
```

### Required Arguments

* `video_file`: Path to the input video
* `--prompt`: Text prompt for SDXL img2img

### Optional Arguments

| Argument               | Description                                            | Default              |
| ---------------------- | ------------------------------------------------------ | -------------------- |
| `--output-path`        | Output directory for generated frames                  | `video_input_frames` |
| `--method`             | Mask replacement method: `none`, `gaussian`, `inpaint` | `none`               |
| `--target-color`       | RGB color to mask                                      | `0 0 0`              |
| `--color-range`        | Tolerance around target color                          | `30`                 |
| `--noise-level`        | Noise strength for gaussian method                     | `50`                 |
| `--inpaint-radius`     | Radius for OpenCV inpainting                           | `30`                 |
| `--width`              | Frame width                                            | `1024`               |
| `--height`             | Frame height                                           | `1024`               |
| `--steps`              | Diffusion steps                                        | `35`                 |
| `--guidance-scale`     | Classifier-free guidance scale                         | `7.5`                |
| `--seed`               | Random seed                                            | `1337970693`         |
| `--denoising-strength` | Img2img strength                                       | `0.6`                |
| `--temporal-strength`  | Influence of previous generated frame (0–1)            | `0.6`                |

---

## Temporal Consistency

Temporal consistency is achieved by merging the previously generated frame with the current input frame before img2img inference.

* Luminance is blended between frames
* Color channels are biased toward the previous frame

Higher `--temporal-strength` values increase stability but may reduce motion responsiveness.

---

## Output

* Frames are saved as PNG files
* Filenames are zero-padded to 16 digits
* Example:

```text
0000000000000000.png
0000000000000001.png
0000000000000002.png
```

These frames can later be stitched back into a video using tools such as ffmpeg.

---

## Notes

* This script is GPU-intensive and optimized for offline batch processing
* Designed for experimentation with video stylization and diffusion-based animation
* Temporal merging happens before diffusion, not inside the model

---

## License

This project is provided as-is for research and experimentation purposes.
