import copy
import os
import os.path as osp
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from poseformerv2.model_poseformer import PoseTransformerV2 as Model
from poseformerv2.camera import *
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'

VIDEO_PATH = Path("/media/Eason/TMU_dataset/stable_usntable/")
STABLE_PATH = VIDEO_PATH / "stable"
UNSTABLE_PATH = VIDEO_PATH / "unstable"
VIDEOS = list(STABLE_PATH.iterdir()) + list(UNSTABLE_PATH.iterdir())

os.makedirs("output2")
SKELETON_OUTPUT_PATH          = "output2/NEW_video_skel.pkl"
SARCOPENIA_LABELS_OUTPUT_PATH = "output2/NEW_sarcopenia_labels.txt"
UNSTABLE_LABELS_OUTPUT_PATH   = "output2/NEW_unstable_labels.txt"
PATHS_OUTPUT_PATH             = "output2/NEW_paths.txt"
SUBJECT_OUTPUT_PATH           = "output2/NEW_subjects.txt"
MEDIAPIPE_MODEL_PATH          = "models/mediapipe/pose_landmarker_heavy.task"

# DEBUG
VIDEOS = VIDEOS[:3]
print (VIDEOS)

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


def video_to_array(model, video_path, args):
    """
    Processes a video to extract 3D skeleton data in NTU-25 format.
    
    1. Runs HRNet to get 2D COCO keypoints.
    2. Converts 2D COCO -> 2D H36M format.
    3. Runs PoseTransformerV2 to get 3D H36M-17 keypoints.
    4. Converts 3D H36M-17 -> 3D NTU-25 keypoints.
    
    Args:
        model: The loaded PoseTransformerV2 model.
        video_path: String path to the input video file.
        args: The argparse object from main() containing model parameters 
              (e.g., args.pad, args.frames).
              
    Returns:
        np.ndarray: Shape (N, 25, 3) representing 3D keypoints for N frames.
    """

    # Helper function to convert the 3D H36M-17 pose to NTU-25
    # This uses the same logic as your coco17_to_ntu25 but for the H36M-17 skeleton
    def _h36m17_to_ntu25(h36m_kps):
        """
        Converts H36M 17 keypoints (shape (17,3)) to NTU 25 keypoints (shape (25,3)).
        Input (h36m_kps) is (x, y, z).
        _safe_mean (defined in global scope) works correctly for (x,y,z).
        """
        h36m_kps = np.asarray(h36m_kps, dtype=float)
        if h36m_kps.shape != (17, 3):
            raise ValueError("h36m_kps must have shape (17,3)")
        ntu = np.full((25, 3), np.nan, dtype=float)

        # H36M-17 Joint Indices:
        # 0:Pelvis, 1:R_Hip, 2:R_Knee, 3:R_Ankle, 4:L_Hip, 5:L_Knee, 6:L_Ankle,
        # 7:Spine, 8:Thorax, 9:Nose, 10:Head, 11:L_Shoulder, 12:L_Elbow,
        # 13:L_Wrist, 14:R_Shoulder, 15:R_Elbow, 16:R_Wrist
        def K(idx):
            return h36m_kps[idx]

        # Compute convenience points
        left_shoulder = K(11)
        right_shoulder = K(14)
        left_hip = K(4)
        right_hip = K(1)
        left_wrist = K(13)
        right_wrist = K(16)
        
        # 1. Torso & Head
        spine_base = K(0)     # Use H36M Pelvis as base of spine
        neck = _safe_mean([left_shoulder, right_shoulder])
        middle_spine = _safe_mean([spine_base, neck]) # Approx
        upper_spine = _safe_mean([middle_spine, neck])  # Approx
        head = K(10)          # Use H36M Head

        ntu[0]  = spine_base
        ntu[1]  = middle_spine
        ntu[2]  = neck
        ntu[3]  = head
        ntu[20] = upper_spine

        # 2. Left Arm (proxies for hand)
        ntu[4]  = left_shoulder
        ntu[5]  = K(12) # L_Elbow
        ntu[6]  = left_wrist
        ntu[7]  = left_wrist  # left hand ~ left wrist
        ntu[21] = left_wrist  # tip of left hand ~ left wrist
        ntu[22] = left_wrist  # left thumb ~ left wrist

        # 3. Right Arm (proxies for hand)
        ntu[8]  = right_shoulder
        ntu[9]  = K(15) # R_Elbow
        ntu[10] = right_wrist
        ntu[11] = right_wrist # right hand ~ right wrist
        ntu[23] = right_wrist # tip of right hand ~ right wrist
        ntu[24] = right_wrist # right thumb ~ right wrist
        
        # 4. Left Leg (proxies for foot)
        ntu[12] = left_hip
        ntu[13] = K(5) # L_Knee
        ntu[14] = K(6) # L_Ankle
        ntu[15] = K(6) # left foot ≈ left ankle

        # 5. Right Leg (proxies for foot)
        ntu[16] = right_hip
        ntu[17] = K(2) # R_Knee
        ntu[18] = K(3) # R_Ankle
        ntu[19] = K(3) # right foot ≈ right ankle

        return ntu

    # === Step 1 & 2: Get 2D H36M-17 Keypoints ===
    # 1. Get 2D keypoints (COCO 17)
    # gen_output=False avoids writing debug files, just returns arrays
    keypoints_coco, scores_coco = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=False)

    # 2. Convert COCO 17 -> H36M 17 (using your provided function)
    # keypoints_h36m shape: (1, N_frames, 17, 2)
    # scores_h36m shape: (1, N_frames, 17)
    keypoints_h36m, scores_h36m, valid_frames = h36m_coco_format(keypoints_coco, scores_coco)
    
    # Get keypoints for the first (and only) person
    # Shape: (N_frames, 17, 2)
    keypoints_2d = keypoints_h36m[0]

    # === Step 3: Get 3D H36M-17 Keypoints ===
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Ensure frame counts match. hrnet_pose might drop a frame.
    # Use the number of detected frames as the source of truth.
    if video_length != len(keypoints_2d):
        # print(f"Warning: Video length mismatch. CV2: {video_length}, HRNet: {len(keypoints_2d)}. Using HRNet length.")
        video_length = len(keypoints_2d)
        
    pad = args.pad
    frames_count = args.frames
    all_3d_poses_h36m = [] # To store (17, 3) poses

    for i in range(video_length):
        # 1. Get the sliding window of 2D keypoints
        start = max(0, i - pad)
        end = min(i + pad, video_length - 1)
        
        input_2D_no = keypoints_2d[start:end+1]

        # 2. Pad the window if it's at the edges
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != frames_count:
            if i < pad:
                left_pad = pad - i
            if i > video_length - pad - 1:
                # Calculate how many frames we're short on the right
                right_pad = (i + pad) - (video_length - 1)
            
            # Pad the array to the required frame count
            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

        # 3. Normalize coordinates
        # This function is from your `poseformerv2.camera` import
        input_2D = normalize_screen_coordinates(input_2D_no, w=width, h=height)

        # 4. Augment (flip)
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
        
        # 5. Batch and format for model
        # Shape: (2, N_frames, 17, 2)
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        input_2D = torch.from_numpy(input_2D.astype('float32')).to("cuda")

        # Add batch dimension for the model
        # Shape: (1, 2, N_frames, 17, 2)
        input_2D = input_2D.unsqueeze(0)

        # 6. Run model
        with torch.no_grad():
            # Input shape: (1, N_frames, 17, 2)
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip = model(input_2D[:, 1])

        # 7. Post-process (average flipped and non-flipped)
        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
        
        output_3D = (output_3D_non_flip + output_3D_flip) / 2
        output_3D[:, :, 0, :] = 0 # Root-relative (set pelvis to 0,0,0)
        
        # Get the 3D pose for the center frame
        # Shape: (17, 3)
        post_out = output_3D[0, 0].cpu().numpy()
        
        all_3d_poses_h36m.append(post_out)

    # Stack all (17, 3) poses into (N, 17, 3)
    all_3d_poses_h36m = np.stack(all_3d_poses_h36m, axis=0)

    # === Step 4: Convert 3D H36M-17 -> 3D NTU-25 ===
    all_3d_poses_ntu = []
    for i in range(all_3d_poses_h36m.shape[0]):
        pose_17 = all_3d_poses_h36m[i]
        pose_25 = _h36m17_to_ntu25(pose_17)
        all_3d_poses_ntu.append(pose_25)
        
    # Stack all (25, 3) poses into (N, 25, 3)
    return np.stack(all_3d_poses_ntu, axis=0)

def main():
    print("Loading model")
    # MODEL_PATH = 'models/poseformer/9_81_46.0.bin'
    # args, _ = argparse.ArgumentParser().parse_known_args()
    # args.embed_dim_ratio, args.depth, args.frames = 32, 4, 81
    # args.number_of_kept_frames, args.number_of_kept_coeffs = 9, 9
    # args.pad = (args.frames - 1) // 2
    # args.previous_dir = 'checkpoint/'
    # args.n_joints, args.out_joints = 17, 17

    MODEL_PATH = 'models/poseformer/27_243_45.2.bin'
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
    args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/'
    args.n_joints, args.out_joints = 17, 17
    
    # model = Model(args=args).to("cuda")
    model = nn.DataParallel(Model(args=args)).cuda()
    model.load_state_dict(torch.load(MODEL_PATH)['model_pos'], strict=True)
    model.eval()

    video_skel = []
    sarcopenia_labels = []
    unstable_labels = []
    subjects = []

    for i, video_path in enumerate(tqdm(VIDEOS)):
        parts = video_path.stem.split("_")
        label_stable_unstable = int(parts[0]) # 0 = stable/1 = unstable
        subject = int(parts[1][:-2]) # there's a trailing "tg" for some reason, we trim it off here (thus the :-2)
        label_sarcopenia_normal = int(parts[2]) # 1 = sarcopenia/2 = normal

        try:
            arr = video_to_array(model, str(video_path), args)
            arr = arr.reshape(-1, 75)
            video_skel.append(arr)
            sarcopenia_labels.append(label_sarcopenia_normal)
            unstable_labels.append(label_stable_unstable)
            subjects.append(subject)
        except Exception as e:
            print(f"Skipping video idx={i}, Caught exception: {e}")

    # Save all the data
    print(f"Saving skeleton data to {SKELETON_OUTPUT_PATH}")
    with open(SKELETON_OUTPUT_PATH, 'wb') as f:
        pickle.dump(video_skel, f)
    print(f"Saving sarcopenia/normal data to {SARCOPENIA_LABELS_OUTPUT_PATH}")
    with open(SARCOPENIA_LABELS_OUTPUT_PATH, 'w') as f:
        f.writelines(f"{label}\n" for label in sarcopenia_labels)
    print(f"Saving stable/unstable label data to {UNSTABLE_LABELS_OUTPUT_PATH}")
    with open(UNSTABLE_LABELS_OUTPUT_PATH, 'w') as f:
        f.writelines(f"{label}\n" for label in unstable_labels)
    print(f"Saving paths data to {PATHS_OUTPUT_PATH}")
    with open(PATHS_OUTPUT_PATH, 'w') as f:
        f.writelines(f"{path}\n" for path in VIDEOS)
    print(f"Saving subject data to {SUBJECT_OUTPUT_PATH}")
    with open(SUBJECT_OUTPUT_PATH, 'w') as f:
        f.writelines(f"{subject}\n" for subject in subjects)

if __name__ == "__main__":
    main()

format
