#! venv/bin/python3

import cv2
import sys
from pathlib import Path

def split_video_to_frames(video_path):
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    output_dir = Path("video_input_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        sys.exit(1)

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = f"{frame_index:016d}.png"
        output_path = output_dir / filename

        cv2.imwrite(str(output_path), frame)
        frame_index += 1

    cap.release()
    print(f"Done. Extracted {frame_index} frames into '{output_dir}/'")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 video_to_frames.py <video_file>")
        sys.exit(1)

    split_video_to_frames(sys.argv[1])

if __name__ == "__main__":
    main()

