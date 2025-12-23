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

# RIFE Folder Stitcher

This script takes a folder of sequential image frames and uses RIFE (Real-Time Intermediate Flow Estimation) to interpolate intermediate frames between every consecutive pair. The result is a temporally smooth, high-framerate frame sequence suitable for final video encoding.

It is designed to work as the second stage of a pipeline where frames are first generated or stylized (for example via SDXL img2img), and then motion smoothness is restored using neural frame interpolation.

---

## What This Script Achieves

Given an input directory containing frames like:

```text
0000000000000000.png
0000000000000001.png
0000000000000002.png
```

The script:

1. Iterates over every consecutive frame pair `(N, N+1)`
2. Runs RIFE interpolation for each pair
3. Collects all generated intermediate frames
4. Renumbers them into a single continuous sequence
5. Writes the result into a final output directory

The final frame set can then be encoded into a smooth video using ffmpeg.

---

## Directory Layout

This script assumes the following project structure:

```text
project_root/
├── video_input_frames/     # Input frames (from SDXL or other source)
├── final_frames/           # Output frames after RIFE interpolation
└── ECCV2022-RIFE/
    ├── inference_img.py    # RIFE inference script
    ├── output/             # Temporary RIFE output (auto-cleaned)
    └── rife-folder-stitch.py
```

Paths are resolved automatically based on the script location.

---

## Requirements

* Python 3.9+
* PyTorch with CUDA support
* A working RIFE repository (ECCV2022-RIFE)
* NVIDIA GPU recommended

The script calls RIFE via `inference_img.py` using the same Python interpreter that launched it.

---

## Configuration

Key configuration values are defined at the top of the script:

| Variable           | Description                                         | Default              |
| ------------------ | --------------------------------------------------- | -------------------- |
| `INPUT_DIR`        | Directory containing input frames                   | `video_input_frames` |
| `FINAL_OUTPUT_DIR` | Directory for stitched output frames                | `final_frames`       |
| `EXP`              | RIFE interpolation exponent (2^EXP frames per pair) | `6`                  |
| `PAD`              | Zero-padding width for frame numbering              | `16`                 |

`EXP=6` generates 64 interpolated steps between each frame pair.

---

## How It Works

### 1. Frame Pair Iteration

The script walks through the input frames in sorted order and processes each consecutive pair.

### 2. RIFE Execution

For each pair, RIFE is executed via:

```bash
python inference_img.py --img frameA.png frameB.png --exp EXP
```

RIFE writes its output into a temporary directory.

### 3. Frame Collection and Renumbering

* All RIFE output frames are collected
* `img0.png` is skipped for all pairs except the first to avoid duplicate frames
* Frames are renumbered into a single global sequence

This guarantees a continuous, duplication-free frame timeline.

---

## Output

The final frames are written to:

```text
final_frames/
```

With filenames like:

```text
0000000000000000.png
0000000000000001.png
0000000000000002.png
```

These frames are ready for direct video encoding.

---

## FFmpeg Example

After completion, the script prints a ready-to-use ffmpeg command:

```bash
ffmpeg -framerate 30 -i final_frames/%016d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4
```

Adjust the framerate as needed depending on your interpolation factor.

---

## Notes

* Temporary RIFE output is cleaned before each interpolation step
* The script fails fast if RIFE produces no frames
* Designed for batch, offline processing
* Works best when input frames are already temporally coherent

---

## Intended Use

This script is ideal for:

* AI-generated animation pipelines
* SDXL or diffusion-based video stylization
* Increasing perceived framerate without re-rendering frames
* Reducing motion stutter in generative video workflows

---

## License

Provided as-is for research and experimentation purposes.

