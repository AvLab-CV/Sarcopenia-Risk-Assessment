import copy
import os.path as osp
import sys
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from poseformerv2.model_poseformer import PoseTransformerV2 as Model
from poseformerv2.camera import normalize_screen_coordinates, camera_to_world
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose

# Prevent downstream argparse users (e.g., HRNet) from seeing PoseFormer CLI args.
# TODO: ????
# ERROR: ???????
# sys.argv = sys.argv[:1]


# ---- Shared geometry helpers (2D COCO -> H36M for PoseFormer) -----------------
h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]


def h36m_coco_format(keypoints, scores):
    assert len(keypoints.shape) == 4 and len(scores.shape) == 3

    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    for i in range(keypoints.shape[0]):
        kpts = keypoints[i]
        score = scores[i]

        new_score = np.zeros_like(score, dtype=np.float32)

        if np.sum(kpts) != 0.0:
            kpts, valid_frame = coco_h36m(kpts)
            h36m_kpts.append(kpts)
            valid_frames.append(valid_frame)

            new_score[:, h36m_coco_order] = score[:, coco_order]
            new_score[:, 0] = np.mean(score[:, [11, 12]], axis=1, dtype=np.float32)
            new_score[:, 8] = np.mean(score[:, [5, 6]], axis=1, dtype=np.float32)
            new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
            new_score[:, 10] = np.mean(score[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

            h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32)

    return h36m_kpts, h36m_scores, valid_frames


def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head, thorax, pelvis, spine
    htps_keypoints[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
    htps_keypoints[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]
    htps_keypoints[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    htps_keypoints[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)
    htps_keypoints[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    keypoints_h36m[:, 9, :] -= (keypoints_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    keypoints_h36m[:, 7, 0] += 2 * (keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]

    return keypoints_h36m, valid_frames


# ---- Backend implementations -------------------------------------------------
class PoseFormerV2Extractor:
    """PoseFormerV2 backend: HRNet 2D -> PoseFormerV2 3D (H36M 17 joints)."""

    def __init__(self, device: torch.device):
        MODEL_PATH = "models/poseformer/27_243_45.2.bin"
        print("Loading PoseFormerV2 model")
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
        args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
        args.pad = (args.frames - 1) // 2
        args.previous_dir = "checkpoint/"
        args.n_joints, args.out_joints = 17, 17

        model = nn.DataParallel(Model(args=args)).to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state["model_pos"], strict=True)
        model.eval()

        self.model = model
        self.device = device
        self.pad = 40
        self.window = 81  # PoseFormer expects 243; we slide with padding.

    def __call__(self, video_path: Path) -> np.ndarray:
        print(f"2D keypoint extraction: {osp.basename(video_path)}")
        keypoints, scores = hrnet_pose(str(video_path), det_dim=416, num_peroson=1, gen_output=True)
        keypoints, scores, _ = h36m_coco_format(keypoints, scores)

        print(f"3D keypoint extraction: {osp.basename(video_path)}")
        cap = cv2.VideoCapture(str(video_path))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        h36m_skeleton_frames: List[np.ndarray] = []
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        for i in tqdm(range(frame_count)):
            img_size = (height, width)
            start = max(0, i - self.pad)
            end = min(i + self.pad, len(keypoints[0]) - 1)
            input_2d_no = keypoints[0][start : end + 1]

            left_pad, right_pad = 0, 0
            if input_2d_no.shape[0] != self.window:
                if i < self.pad:
                    left_pad = self.pad - i
                if i > len(keypoints[0]) - self.pad - 1:
                    right_pad = i + self.pad - (len(keypoints[0]) - 1)

                input_2d_no = np.pad(input_2d_no, ((left_pad, right_pad), (0, 0), (0, 0)), "edge")

            input_2d = normalize_screen_coordinates(input_2d_no, w=img_size[1], h=img_size[0])

            input_2d_aug = copy.deepcopy(input_2d)
            input_2d_aug[:, :, 0] *= -1
            input_2d_aug[:, joints_left + joints_right] = input_2d_aug[:, joints_right + joints_left]
            input_2d = np.concatenate(
                (np.expand_dims(input_2d, axis=0), np.expand_dims(input_2d_aug, axis=0)),
                axis=0,
            )
            input_2d = input_2d[np.newaxis, :, :, :, :]
            input_2d = torch.from_numpy(input_2d.astype("float32")).to(self.device)

            output_3d_non_flip = self.model(input_2d[:, 0])
            output_3d_flip = self.model(input_2d[:, 1])

            output_3d_flip[:, :, :, 0] *= -1
            output_3d_flip[:, :, joints_left + joints_right, :] = output_3d_flip[:, :, joints_right + joints_left, :]
            output_3d = (output_3d_non_flip + output_3d_flip) / 2

            output_3d[:, :, 0, :] = 0
            post_out = output_3d[0, 0].cpu().detach().numpy()

            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype="float32")
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            h36m_skeleton_frames.append(post_out)

        return np.stack(h36m_skeleton_frames)


class MediaPipeExtractor:
    """MediaPipe BlazePose backend (33 joints, world coordinates)."""

    def __init__(self, model_path: Path, use_gpu: bool):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        if not osp.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model file not found at {model_path}. "
                "Download from https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            )

        self.mp = mp
        base_options = python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def __call__(self, video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        world_landmarks: List[np.ndarray] = []
        for frame_idx in tqdm(range(frame_count), desc=f"Mediapipe {video_path.name}"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(1000 * frame_idx / fps)
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.pose_world_landmarks:
                person_landmarks = detection_result.pose_world_landmarks[0]
                frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in person_landmarks], dtype=np.float32)
            else:
                frame_landmarks = np.zeros((33, 3), dtype=np.float32)
            world_landmarks.append(frame_landmarks)

        cap.release()
        return np.stack(world_landmarks)


class MetrabsExtractor:
    """MeTRAbs backend (H36M 17 joints)."""

    def __init__(self, batch_size: int = 8, skeleton: str = "h36m_17"):
        import tensorflow as tf
        import tensorflow_hub as tfhub
        import cameralib

        self.tf = tf
        self.cameralib = cameralib
        self.model = tfhub.load("https://bit.ly/metrabs_l")
        self.batch_size = batch_size
        self.skeleton = skeleton

    def __call__(self, video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        if not frames:
            return np.zeros((0, 17, 3), dtype=np.float32)

        height, width = frames[0].shape[:2]
        camera = self.cameralib.Camera.from_fov(fov_degrees=55, imshape=(height, width))
        intrinsics = self.tf.convert_to_tensor(camera.intrinsic_matrix, dtype=self.tf.float32)[self.tf.newaxis, ...]

        skeleton_frames: List[np.ndarray] = []
        for start in range(0, len(frames), self.batch_size):
            batch_np = np.stack(frames[start : start + self.batch_size])
            batch = self.tf.convert_to_tensor(batch_np, dtype=self.tf.uint8)
            intrinsics_batch = self.tf.repeat(intrinsics, repeats=batch.shape[0], axis=0)
            predictions = self.model.detect_poses_batched(
                batch, intrinsic_matrix=intrinsics_batch, skeleton=self.skeleton
            )
            poses3d = predictions["poses3d"].numpy()  # (batch, people, joints, 3)

            for pose_set in poses3d:
                if pose_set.shape[0] == 0:
                    skeleton_frames.append(np.zeros((17, 3), dtype=np.float32))
                else:
                    skeleton_frames.append(np.asarray(pose_set[0], dtype=np.float32))

        return np.stack(skeleton_frames)


# ---- CLI / Runner ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Extract 3D skeletons from videos with pluggable backends.")
    parser.add_argument("video_dir_path", type=Path, help="Directory containing MP4 videos.")
    parser.add_argument("output_path", type=Path, help="Output .npz to store skeletons.")
    parser.add_argument(
        "--backend",
        choices=["poseformerv2", "mediapipe", "metrabs"],
        default="poseformerv2",
        help="Pose estimation backend to use.",
    )
    parser.add_argument("--limit", type=int, help="Optionally limit the number of videos processed.")
    parser.add_argument(
        "--mediapipe-model-path",
        type=Path,
        default=Path("models/mediapipe/pose_landmarker_heavy.task"),
        help="Path to MediaPipe pose_landmarker task file.",
    )
    parser.add_argument(
        "--mediapipe-use-gpu",
        action="store_true",
        help="Use GPU delegate for MediaPipe (if available).",
    )
    parser.add_argument(
        "--metrabs-batch-size",
        type=int,
        default=8,
        help="Batch size for MeTRAbs inference.",
    )
    return parser.parse_args()


def build_extractor(args) -> Tuple[object, int]:
    """Return backend callable and joint count (for flattening)."""
    if args.backend == "poseformerv2":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = PoseFormerV2Extractor(device=device)
        return extractor, 17
    if args.backend == "mediapipe":
        extractor = MediaPipeExtractor(model_path=args.mediapipe_model_path, use_gpu=args.mediapipe_use_gpu)
        return extractor, 33
    if args.backend == "metrabs":
        extractor = MetrabsExtractor(batch_size=args.metrabs_batch_size, skeleton="smpl")
        return extractor, 24

    raise NotImplementedError("Choose a backend")

def main():
    args = parse_args()

    video_path: Path = args.video_dir_path
    output_path: Path = args.output_path
    videos = list(video_path.iterdir())
    videos = [v for v in videos if v.is_file()]

    # Limit to N samples
    if args.limit:
        videos = videos[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor, joint_count = build_extractor(args)
    skel_by_video: Dict[str, np.ndarray] = {}

    for i, video in enumerate(tqdm(videos, desc="Videos")):
        try:
            frames = extractor(video)  # (frame, joints, 3)
            frames = frames.reshape(frames.shape[0], joint_count * 3)
            skel_by_video[video.name] = frames
        except Exception as e:
            print(f"Skipping video idx={i}, {video.name}, caught exception: {e}")

    print(f"Saving skeleton data to {output_path}")
    np.savez(output_path, **skel_by_video)


if __name__ == "__main__":
    main()

