import argparse
from pathlib import Path
from typing import Callable, Dict, Tuple

import einops
import numpy as np


H36M = {
    "pelvis": 0,
    "right_hip": 1,
    "right_knee": 2,
    "right_ankle": 3,
    "left_hip": 4,
    "left_knee": 5,
    "left_ankle": 6,
    "spine": 7,
    "thorax": 8,
    "neck": 9,
    "head": 10,
    "left_shoulder": 11,
    "left_elbow": 12,
    "left_wrist": 13,
    "right_shoulder": 14,
    "right_elbow": 15,
    "right_wrist": 16,
}


def reshape_frames(arr: np.ndarray, expected_joints: int) -> np.ndarray:
    """Ensure shape is (frames, joints, 3)."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1:] == (expected_joints, 3):
        return arr
    if arr.ndim == 2 and arr.shape[1] == expected_joints * 3:
        return arr.reshape(-1, expected_joints, 3)
    if arr.ndim == 2 and arr.shape == (expected_joints, 3):
        return arr[np.newaxis, ...]
    raise ValueError(f"Expected joints={expected_joints}, got shape {arr.shape}")


def h36m17_to_ntu25_frame(h36m_kps: np.ndarray) -> np.ndarray:
    """Convert a single frame of H36M (17x3) to NTU (25x3)."""
    if h36m_kps.shape != (17, 3):
        raise ValueError("h36m_kps must have shape (17, 3)")

    ntu = np.zeros((25, 3), dtype=float)

    def J(name):
        return h36m_kps[H36M[name]]

    pelvis = J("pelvis")
    spine = J("spine")
    thorax = J("thorax")
    neck = J("neck")
    head = J("head")
    left_wrist = J("left_wrist")
    right_wrist = J("right_wrist")
    left_ankle = J("left_ankle")
    right_ankle = J("right_ankle")

    ntu[0] = pelvis
    ntu[1] = spine
    ntu[2] = neck
    ntu[3] = head

    ntu[4] = J("left_shoulder")
    ntu[5] = J("left_elbow")
    ntu[6] = left_wrist
    ntu[7] = left_wrist

    ntu[8] = J("right_shoulder")
    ntu[9] = J("right_elbow")
    ntu[10] = right_wrist
    ntu[11] = right_wrist

    ntu[12] = J("left_hip")
    ntu[13] = J("left_knee")
    ntu[14] = left_ankle
    ntu[15] = left_ankle

    ntu[16] = J("right_hip")
    ntu[17] = J("right_knee")
    ntu[18] = right_ankle
    ntu[19] = right_ankle

    ntu[20] = thorax

    ntu[21] = left_wrist
    ntu[22] = left_wrist
    ntu[23] = right_wrist
    ntu[24] = right_wrist

    return ntu


def blazepose33_to_ntu25_frame(mediapipe_kpts: np.ndarray) -> np.ndarray:
    """Convert a single frame of MediaPipe BlazePose (33x3) to NTU (25x3)."""
    if mediapipe_kpts.shape != (33, 3):
        raise ValueError("mediapipe_kpts must have shape (33, 3)")

    ntu_kpts = np.zeros((25, 3), dtype=np.float32)

    mid_hip = (mediapipe_kpts[23] + mediapipe_kpts[24]) / 2
    mid_shoulder = (mediapipe_kpts[11] + mediapipe_kpts[12]) / 2

    ntu_kpts[0] = mid_hip
    ntu_kpts[1] = (mid_hip + mid_shoulder) / 2
    ntu_kpts[2] = mid_shoulder
    ntu_kpts[3] = mediapipe_kpts[0]

    ntu_kpts[4] = mediapipe_kpts[11]
    ntu_kpts[5] = mediapipe_kpts[13]
    ntu_kpts[6] = mediapipe_kpts[15]
    ntu_kpts[7] = mediapipe_kpts[19]

    ntu_kpts[8] = mediapipe_kpts[12]
    ntu_kpts[9] = mediapipe_kpts[14]
    ntu_kpts[10] = mediapipe_kpts[16]
    ntu_kpts[11] = mediapipe_kpts[20]

    ntu_kpts[12] = mediapipe_kpts[23]
    ntu_kpts[13] = mediapipe_kpts[25]
    ntu_kpts[14] = mediapipe_kpts[27]
    ntu_kpts[15] = mediapipe_kpts[31]

    ntu_kpts[16] = mediapipe_kpts[24]
    ntu_kpts[17] = mediapipe_kpts[26]
    ntu_kpts[18] = mediapipe_kpts[28]
    ntu_kpts[19] = mediapipe_kpts[32]

    ntu_kpts[20] = ntu_kpts[1]

    ntu_kpts[21] = mediapipe_kpts[17]
    ntu_kpts[22] = mediapipe_kpts[21]
    ntu_kpts[23] = mediapipe_kpts[18]
    ntu_kpts[24] = mediapipe_kpts[22]

    return ntu_kpts


def rotate_x(frames: np.ndarray, degrees: float = 100.0) -> np.ndarray:
    """Rotate frames around the X-axis by the given degrees."""
    theta = np.deg2rad(degrees)
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return einops.einsum(R, frames, "row col, frame joint row -> frame joint col")


def converter_for_input(input_format: str) -> Tuple[Callable[[np.ndarray], np.ndarray], int]:
    if input_format == "h36m17":
        return h36m17_to_ntu25_frame, 17
    if input_format == "blazepose33":
        return blazepose33_to_ntu25_frame, 33
    raise ValueError(f"Unsupported input format {input_format}")


def main():
    parser = argparse.ArgumentParser(description="Convert predicted skeletons to NTU25.")
    parser.add_argument("input", type=Path, help="Input .npz with per-video skeleton arrays.")
    parser.add_argument("output", type=Path, help="Output .npz with NTU25 skeletons.")
    parser.add_argument(
        "--input-format",
        required=True,
        choices=["h36m17", "blazepose33"],
        help="Joint format of the input skeletons.",
    )
    parser.add_argument(
        "--skip-rotation",
        action="store_true",
        help="Skip the default 100-degree X-axis rotation.",
    )
    args = parser.parse_args()

    print(f"Loading from {args.input}")
    skels: Dict[str, np.ndarray] = dict(np.load(args.input))

    convert_frame, expected_joints = converter_for_input(args.input_format)

    for k, arr in skels.items():
        frames = reshape_frames(arr, expected_joints)
        frames = np.stack([convert_frame(frame) for frame in frames])
        if not args.skip_rotation:
            frames = rotate_x(frames)
        skels[k] = frames.reshape(frames.shape[0], -1)

    print(f"Saving to {args.output}")
    np.savez(args.output, **skels)


if __name__ == "__main__":
    main()
