#!venv-rife/bin/python3
import subprocess
import shutil
from pathlib import Path
import sys
import time
import re

# =========================
# CONFIG
# =========================

RIFE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RIFE_ROOT.parent

INPUT_DIR = PROJECT_ROOT / "video_input_frames"
RIFE_SCRIPT = RIFE_ROOT / "inference_img.py"
RIFE_OUTPUT_DIR = RIFE_ROOT / "output"
FINAL_OUTPUT_DIR = PROJECT_ROOT / "final_frames"

EXP = 5
PYTHON_BIN = sys.executable

PAD = 16  # match SDXL frame numbering

# =========================
# HELPERS
# =========================

def log(msg):
    print(f"[RIFE-STITCH] {msg}", flush=True)

def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def run_rife(img_a: Path, img_b: Path):
    log("Interpolating:")
    log(f"  A = {img_a.name}")
    log(f"  B = {img_b.name}")

    cmd = [
        PYTHON_BIN,
        str(RIFE_SCRIPT),
        "--img",
        str(img_a),
        str(img_b),
        "--exp",
        str(EXP),
    ]

    result = subprocess.run(
        cmd,
        cwd=RIFE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("RIFE failed")

def collect_and_renumber(start_index: int) -> int:
    frames = sorted(
        RIFE_OUTPUT_DIR.glob("img*.png"),
        key=lambda p: int(re.findall(r"\d+", p.stem)[0])
    )

    if len(frames) < 3:
        raise RuntimeError("RIFE output too small to contain interpolated frames")

    # Drop endpoints: img0 (FrameA) and imgN (FrameB)
    interpolated = frames[1:-1]

    for frame in interpolated:
        out_name = f"{start_index:0{PAD}d}.png"
        dest = FINAL_OUTPUT_DIR / out_name
        shutil.move(frame, dest)
        start_index += 1

    return start_index

# =========================
# MAIN
# =========================

def main():
    log(f"RIFE_ROOT    = {RIFE_ROOT}")
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"INPUT_DIR    = {INPUT_DIR}")
    log(f"EXP          = {EXP}")

    if not RIFE_SCRIPT.exists():
        raise RuntimeError("inference_img.py not found")

    input_frames = sorted(INPUT_DIR.glob("*.png"))
    if len(input_frames) < 2:
        raise RuntimeError("Need at least two input frames")

    log(f"Found {len(input_frames)} input frames")

    clean_dir(FINAL_OUTPUT_DIR)
    clean_dir(RIFE_OUTPUT_DIR)

    global_index = 0

    for i in range(len(input_frames) - 1):
        log(f"\n=== Pair {i} / {len(input_frames) - 2} ===")

        clean_dir(RIFE_OUTPUT_DIR)

        run_rife(input_frames[i], input_frames[i + 1])

        time.sleep(0.1)

        global_index = collect_and_renumber(global_index)

    log("\nALL DONE")
    log(f"Total interpolated frames written: {global_index}")
    log(f"Final frames dir: {FINAL_OUTPUT_DIR}")

    print("\nFFmpeg:")
    print(
        f"ffmpeg -framerate 30 -i {FINAL_OUTPUT_DIR}/%0{PAD}d.png "
        "-c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4"
    )

if __name__ == "__main__":
    main()

