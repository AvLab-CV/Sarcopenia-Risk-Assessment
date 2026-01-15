import subprocess
from pathlib import Path

# =========================
# CONFIGURATION CONSTANTS
# =========================
INPUT_DIR = Path("/media/Eason/TMU_dataset/tandemgait_full_164")
OUTPUT_DIR = Path("/media/Eason/TMU_dataset/tandemgait_full_164_resized700x700")

TARGET_WIDTH = 700
TARGET_HEIGHT = 700

FFMPEG_BIN = "ffmpeg"
CRF = 23
PRESET = "veryfast"
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing videos in {INPUT_DIR}...")

    for input_file in sorted(INPUT_DIR.glob("*.mp4")):
        output_file = OUTPUT_DIR / input_file.name

        scale_expr = (
            f"scale='min({TARGET_WIDTH}/iw,{TARGET_HEIGHT}/ih)*iw':"
            f"'min({TARGET_WIDTH}/iw,{TARGET_HEIGHT}/ih)*ih',"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
        )

        cmd = [
            FFMPEG_BIN,
            "-i", str(input_file),
            "-vf", scale_expr,
            "-c:v", "libx264",
            "-crf", str(CRF),
            "-preset", PRESET,
            "-an",
            "-y",
            str(output_file)
        ]

        subprocess.run(cmd, check=True)
        print(f"   -> Saved {output_file.name}")

    print(f"✅ All videos resized and saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
