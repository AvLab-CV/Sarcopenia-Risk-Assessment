# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import pickle
from tqdm import tqdm

INPUT          =             './output/output4/part2.pkl'
OUTPUT         = './output/output4/skateformer/part2.npz'

def remove_nan_frames(ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('<skename>\t{:^5}\t{}'.format(f + 1, nan_indices))

    return ske_joints[valid_frames]

def seq_translation(skes_joints):
    for idx, ske_joints in tqdm(enumerate(skes_joints), total=len(skes_joints)):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def align_frames(skes_joints):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = 1000
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 2))

    for idx,  class_idx in enumerate(labels):
        labels_vector[idx, int(class_idx)] = 1

    return labels_vector


if __name__ == '__main__':
    with open(INPUT, 'rb') as fr:
        fold = pickle.load(fr)

    data = {}
    for set_name in ["train", "val", "test"]:
        # DeMorgan's Law.
        if (set_name + "_X") not in fold or (set_name + "_Y") not in fold:
            continue

        print(f"Set: {set_name}")

        skes_joints = fold[set_name + "_X"]
        labels      = fold[set_name + "_Y"]
        skes_joints = seq_translation(skes_joints)
        skes_len = np.array([sk.shape[0] for sk in skes_joints]) # <- save the original length before alignment
        skes_joints = align_frames(skes_joints)  # aligned to the same frame length

        data["x_" + set_name] = skes_joints
        data["len_" + set_name] = skes_len
        data["y_" + set_name] = one_hot_vector(labels)

        print("x shape", data["x_" + set_name].shape)
        print("y shape", data["y_" + set_name].shape)
        print("normal     count=", data["y_" + set_name][:, 0].sum().item())
        print("sarcopenia count=", data["y_" + set_name][:, 1].sum().item())
        print()

    np.savez(OUTPUT, **data)
    print(f"Wrote final training-ready .npz to {OUTPUT}")
