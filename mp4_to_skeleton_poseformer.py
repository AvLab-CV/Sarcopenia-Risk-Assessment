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

def h36m17_to_ntu25(h36m_kps):
    """
    Convert Human3.6M 3D joints (17 x 3) to the NTU RGB+D (25 x 3) format.
    PoseFormer predicts joints in Human3.6M order, so we map them directly
    instead of treating them like COCO 2D detections.
    """
    h36m_kps = np.asarray(h36m_kps, dtype=float)
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
    ntu[7] = left_wrist  # no dedicated hand joints

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

    h36m_skeleton_frames = []

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
        h36m_skeleton_frames.append(post_out)


    ntu_skeleton_frames = [h36m17_to_ntu25(kpts) for kpts in h36m_skeleton_frames]
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

