import cv2
from pathlib import Path

def save_frames(video_path, frame_indices, output_dir="frames"):
    """
    Extract and save specific frames from a video as PNG files.
    
    Args:
        video_path: Path to the video file
        frame_indices: List of frame indices to extract (0-based)
        output_dir: Directory to save the frames (default: "frames")
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for idx in sorted(frame_indices):
        if idx >= total_frames:
            print(f"Warning: Frame {idx} exceeds video length ({total_frames} frames)")
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            x = 320
            y = 50
            w = 320
            h = 640
            frame = frame[y:y+h, x:x+w]
            output_path = Path(output_dir) / f"frame_{idx:06d}.png"
            cv2.imwrite(str(output_path), frame)
            print(f"Saved frame {idx} to {output_path}")
        else:
            print(f"Failed to read frame {idx}")
    
    cap.release()


# Example usage
if __name__ == "__main__":
    IN = Path("/Users/aldo/Code/avlab/dataset/merged/2_022.mp4")
    OUT = Path("/Users/aldo/Code/avlab/poseopsfinal/output/figures_vid/")
    OUT.mkdir(exist_ok=True)
    # frame_indices = list(range(0, 90, 15))
    T = 250
    frame_indices = list(range(0, T, 30))
    save_frames(IN, frame_indices, output_dir=OUT)
