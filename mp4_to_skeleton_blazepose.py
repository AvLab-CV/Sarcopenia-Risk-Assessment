import os
import os.path as osp
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from tqdm import tqdm
import pickle

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_PLATFORM'] = 'surfaceless'

GPU = True
# VIDEO_PATH = Path("/media/Eason/TMU_dataset/stable_usntable/")
VIDEO_PATH = Path("/Users/aldo/Code/avlab/dataset/stable_unstable")
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
# VIDEOS = VIDEOS[:3]
# print (VIDEOS)

base_timestamp = 0

def convert_mediapipe_to_ntu25(mediapipe_kpts):
    # This mapping is based on standard skeleton definitions.
    # NTU Joints:
    # 1: base of the spine, 2: middle of the spine, 3: neck, 4: head,
    # 5: left shoulder, 6: left elbow, 7: left wrist, 8: left hand,
    # 9: right shoulder, 10: right elbow, 11: right wrist, 12: right hand,
    # 13: left hip, 14: left knee, 15: left ankle, 16: left foot,
    # 17: right hip, 18: right knee, 19: right ankle, 20: right foot,
    # 21: spine, 22: tip of the left hand, 23: left thumb,
    # 24: tip of the right hand, 25: right thumb
    #
    # MediaPipe BlazePose Joints:
    # We use indices 11, 12, 23, 24 for shoulder/hip centers.

    ntu_kpts = np.zeros((25, 3), dtype=np.float32)

    # Midpoints for spine and neck
    mid_hip = (mediapipe_kpts[23] + mediapipe_kpts[24]) / 2
    mid_shoulder = (mediapipe_kpts[11] + mediapipe_kpts[12]) / 2

    # 1: base of spine (mid_hip)
    ntu_kpts[0] = mid_hip
    # 2: middle of spine (average of mid_hip and mid_shoulder)
    ntu_kpts[1] = (mid_hip + mid_shoulder) / 2
    # 3: neck
    ntu_kpts[2] = mid_shoulder
    # 4: head (use nose)
    ntu_kpts[3] = mediapipe_kpts[0]

    # Left Arm
    ntu_kpts[4] = mediapipe_kpts[11]  # 5: left shoulder
    ntu_kpts[5] = mediapipe_kpts[13]  # 6: left elbow
    ntu_kpts[6] = mediapipe_kpts[15]  # 7: left wrist
    ntu_kpts[7] = mediapipe_kpts[19]  # 8: left hand (use left_index)

    # Right Arm
    ntu_kpts[8] = mediapipe_kpts[12]  # 9: right shoulder
    ntu_kpts[9] = mediapipe_kpts[14]  # 10: right elbow
    ntu_kpts[10] = mediapipe_kpts[16]  # 11: right wrist
    ntu_kpts[11] = mediapipe_kpts[20]  # 12: right hand (use right_index)

    # Left Leg
    ntu_kpts[12] = mediapipe_kpts[23]  # 13: left hip
    ntu_kpts[13] = mediapipe_kpts[25]  # 14: left knee
    ntu_kpts[14] = mediapipe_kpts[27]  # 15: left ankle
    ntu_kpts[15] = mediapipe_kpts[31]  # 16: left foot (use left_foot_index)

    # Right Leg
    ntu_kpts[16] = mediapipe_kpts[24]  # 17: right hip
    ntu_kpts[17] = mediapipe_kpts[26]  # 18: right knee
    ntu_kpts[18] = mediapipe_kpts[28]  # 19: right ankle
    ntu_kpts[19] = mediapipe_kpts[32]  # 20: right foot (use right_foot_index)

    # 21: spine (same as middle of spine)
    ntu_kpts[20] = ntu_kpts[1]

    # Hand tips and thumbs
    ntu_kpts[21] = mediapipe_kpts[17]  # 22: tip of the left hand (use left_pinky)
    ntu_kpts[22] = mediapipe_kpts[21]  # 23: left thumb
    ntu_kpts[23] = mediapipe_kpts[18]  # 24: tip of the right hand (use right_pinky)
    ntu_kpts[24] = mediapipe_kpts[22]  # 25: right thumb

    return ntu_kpts

def video_to_array(landmarker, path):
    global base_timestamp

    print(f"Processing video: {osp.basename(path)}")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_world_landmarks = []
    thisbase = base_timestamp

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to MediaPipe's Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Calculate timestamp for the frame
        timestamp_ms = thisbase + int(1000 * frame_idx / fps)
        base_timestamp = thisbase + int(1000 * (frame_idx + 1) / fps)

        # Detect pose landmarks
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if detection_result.pose_world_landmarks:
            # We only track the first detected person
            person_landmarks = detection_result.pose_world_landmarks[0]

            # Extract XYZ coordinates into a NumPy array
            # The origin (0,0,0) is approximately the center of the hips.
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in person_landmarks])
            all_world_landmarks.append(frame_landmarks)
        else:
            # If no person is detected, append a placeholder of zeros
            # MediaPipe BlazePose model has 33 keypoints
            all_world_landmarks.append(np.zeros((33, 3), dtype=np.float32))

    cap.release()

    ntu_skeletons = [convert_mediapipe_to_ntu25(kpts) for kpts in all_world_landmarks]
    return np.stack(ntu_skeletons)

def main():
    if not osp.exists(MEDIAPIPE_MODEL_PATH):
        print(f"Model file not found at {MEDIAPIPE_MODEL_PATH}")
        print("Please download it from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        return

    print("Initializing MediaPipe Pose Landmarker...")

    if GPU:
        base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH, delegate=python.BaseOptions.Delegate.GPU)
    else:
        base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,  # Not needed for this task
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    video_skel = []
    sarcopenia_labels = []
    unstable_labels = []
    subjects = []

    for i, video_path in enumerate(tqdm(VIDEOS)):
        parts = video_path.stem.split("_")
        label_stable_unstable = int(parts[0]) # 0 = stable/1 = unstable
        subject = int(parts[1][:-2]) # there's a trailing "tg" for some reason, we trim it off here (thus the :-2)
        label_sarcopenia_normal = int(parts[2]) # 0 = sarcopenia/1 = normal

        try:
            arr = video_to_array(landmarker, video_path)
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

