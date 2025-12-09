import copy
import os
import os.path as osp
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from poseformerv2.model_poseformer import PoseTransformerV2 as Model
from poseformerv2.camera import *
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose

parser = argparse.ArgumentParser()
parser.add_argument(
    "video_dir_path",
    type=Path,
)
parser.add_argument(
    "output_path",
    type=Path,
)
parser.add_argument(
    "--limit",
    type=int,
)
args = parser.parse_args()

# Prevent downstream argparse users (e.g., HRNet) from seeing PoseFormer CLI args.
sys.argv = sys.argv[:1]

VIDEO_PATH: Path = args.video_dir_path
SKELETON_OUTPUT_PATH: Path = args.output_path
VIDEOS = list(VIDEO_PATH.iterdir())
SKELETON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Limit to N samples
if args.limit:
    VIDEOS = VIDEOS[:args.limit]

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

        if np.sum(kpts) != 0.:
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
    keypoints_h36m[:, 7, 0] += 2*(keypoints_h36m[:, 7, 0] - np.mean(keypoints_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1])*2/3

    # half body: the joint of ankle and knee equal to hip
    # keypoints_h36m[:, [2, 3]] = keypoints_h36m[:, [1, 1]]
    # keypoints_h36m[:, [5, 6]] = keypoints_h36m[:, [4, 4]]

    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    
    return keypoints_h36m, valid_frames


# indices for COCO (for clarity)
COCO = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

def _safe_mean(points):
    """points: list of (3,) arrays. Returns (3,) mean; ignores NaNs."""
    if len(points) == 0:
        return np.array([np.nan, np.nan, 0.0], dtype=float)
    arr = np.stack(points, axis=0).astype(float)
    # for coordinates: mean ignoring rows that are all-nans
    coords = arr[:, :2]
    confs = arr[:, 2]
    # mask rows where coords are nan
    valid_row = ~np.isnan(coords).all(axis=1)
    if not valid_row.any():
        return np.array([np.nan, np.nan, float(np.nan)])
    mean_xy = np.nanmean(coords[valid_row], axis=0)
    # for confidence we use mean of valid confs
    mean_conf = float(np.nanmean(confs[valid_row]))
    return np.array([mean_xy[0], mean_xy[1], mean_conf], dtype=float)

def coco17_to_ntu25(coco_kps):
    """
    Convert COCO 17 keypoints (shape (17,3)) to NTU 25 keypoints (shape (25,3)).
    Args:
        coco_kps: np.ndarray of shape (17,3) (x, y, confidence/visibility). May contain np.nan for missing coords.
    Returns:
        ntu_kps: np.ndarray shape (25,3) with the NTU joint order (0..24).
    Notes:
        - Where NTU has joints that COCO doesn't (hand tips / thumbs / extra spine points),
          we approximate by averaging or copying nearby keypoints.
        - This mapping is heuristic; tweak if you need different proxies (e.g. use head bbox to estimate hand tips).
    """
    coco_kps = np.asarray(coco_kps, dtype=float)
    if coco_kps.shape != (17, 3):
        raise ValueError("coco_kps must have shape (17,3)")
    ntu = np.full((25, 3), np.nan, dtype=float)

    # helper to fetch COCO joint as array
    def K(name):
        idx = COCO[name]
        return coco_kps[idx]

    # compute some convenience points
    left_shoulder = K("left_shoulder")
    right_shoulder = K("right_shoulder")
    left_hip = K("left_hip")
    right_hip = K("right_hip")
    left_wrist = K("left_wrist")
    right_wrist = K("right_wrist")
    nose = K("nose")

    # Neck ≈ midpoint of shoulders
    neck = _safe_mean([left_shoulder, right_shoulder])

    # Base of spine ≈ midpoint of hips
    spine_base = _safe_mean([left_hip, right_hip])

    # Middle spine ≈ midpoint of base spine and neck
    middle_spine = _safe_mean([spine_base, neck])

    # Upper spine (NTU "spine") ≈ midpoint of middle_spine and neck
    upper_spine = _safe_mean([middle_spine, neck])

    # Head: use nose (COCO has no explicit 'head top'); optionally could average nose + eyes
    head = _safe_mean([nose, K("left_eye"), K("right_eye")])

    # Map NTU indices (0-based) -> description:
    # 0 base of the spine, 1 middle of the spine, 2 neck, 3 head,
    # 4 left shoulder, 5 left elbow, 6 left wrist, 7 left hand,
    # 8 right shoulder, 9 right elbow, 10 right wrist, 11 right hand,
    # 12 left hip, 13 left knee, 14 left ankle, 15 left foot,
    # 16 right hip, 17 right knee, 18 right ankle, 19 right foot,
    # 20 spine (upper), 21 tip of left hand, 22 left thumb, 23 tip of right hand, 24 right thumb

    # Fill direct mappings
    ntu[0]  = spine_base                 # base of spine
    ntu[1]  = middle_spine               # middle of spine
    ntu[2]  = neck                       # neck
    ntu[3]  = head                       # head
    ntu[4]  = left_shoulder
    ntu[5]  = K("left_elbow")
    ntu[6]  = left_wrist
    ntu[7]  = left_wrist                 # left hand ~ left wrist (no hand tip in COCO)
    ntu[8]  = right_shoulder
    ntu[9]  = K("right_elbow")
    ntu[10] = right_wrist
    ntu[11] = right_wrist                # right hand ~ right wrist
    ntu[12] = left_hip
    ntu[13] = K("left_knee")
    ntu[14] = K("left_ankle")
    ntu[15] = K("left_ankle")            # left foot ≈ left ankle (no separate foot in COCO)
    ntu[16] = right_hip
    ntu[17] = K("right_knee")
    ntu[18] = K("right_ankle")
    ntu[19] = K("right_ankle")           # right foot ≈ right ankle
    ntu[20] = upper_spine

    # hand tips & thumbs (21..24) — no direct COCO equivalent: copy wrist (or optionally set NaN)
    ntu[21] = left_wrist   # tip of left hand
    ntu[22] = left_wrist   # left thumb
    ntu[23] = right_wrist  # tip of right hand
    ntu[24] = right_wrist  # right thumb

    # Post-process: if any entry has NaN coords but has non-zero conf in inputs, ensure conf is meaningful.
    # Here we keep computed confidences from _safe_mean / originals; leave NaNs if coords NaN.
    return ntu


def video_to_array(model, video_path):
    PAD = 40
    FRAMES = 81 # hardcoded or use framecount

    print(f"2D keypoint extraction: {osp.basename(video_path)}")
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    print(f"3D keypoint extraction: {osp.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    coco_skeleton_frames = []

    for i in tqdm(range(frame_count)):
        img_size = height, width

        ## input frames
        start = max(0, i - PAD)
        end =  min(i + PAD, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != FRAMES:
            if i < PAD:
                left_pad = PAD - i
            if i > len(keypoints[0]) - PAD - 1:
                right_pad = i + PAD - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        # input_2D_no += np.random.normal(loc=0.0, scale=5, size=input_2D_no.shape)
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        # (2, 243, 17, 2)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## estimation
        output_3D_non_flip = model(input_2D[:, 0]) 
        output_3D_flip     = model(input_2D[:, 1])
        # [1, 1, 17, 3]

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        coco_skeleton_frames.append(post_out)


    ntu_skeleton_frames = [coco17_to_ntu25(kpts) for kpts in coco_skeleton_frames]
    ntu_skeleton_frames = np.stack(ntu_skeleton_frames)
    return ntu_skeleton_frames

def main():
    print("Loading model")
    MODEL_PATH = 'models/poseformer/27_243_45.2.bin'
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
    args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/'
    args.n_joints, args.out_joints = 17, 17
    
    model = nn.DataParallel(Model(args=args)).cuda()
    model.load_state_dict(torch.load(MODEL_PATH)['model_pos'], strict=True)
    model.eval()

    skel_by_video = {}

    for i, video_path in enumerate(tqdm(VIDEOS)):
        try:
            arr = video_to_array(model, str(video_path))
            arr = arr.reshape(-1, 75)
            skel_by_video[video_path.name] = arr
        except Exception as e:
            print(f"Skipping video idx={i}, Caught exception: {e}")

    # Save all the data
    print(f"Saving skeleton data to {SKELETON_OUTPUT_PATH}")
    np.savez(SKELETON_OUTPUT_PATH, **skel_by_video)

if __name__ == "__main__":
    main()

