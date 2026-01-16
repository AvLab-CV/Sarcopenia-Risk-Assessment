import numpy as np
import random

from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', data_type='j',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.load_data()
        if partition:
            self.right_arm = np.array([7, 8, 22, 23]) - 1
            self.left_arm = np.array([11, 12, 24, 25]) - 1
            self.right_leg = np.array([13, 14, 15, 16]) - 1
            self.left_leg = np.array([17, 18, 19, 20]) - 1
            self.h_torso = np.array([5, 9, 6, 10]) - 1
            self.w_torso = np.array([2, 3, 1, 4]) - 1
            self.new_idx = np.concatenate((self.right_arm, self.left_arm, self.right_leg, self.left_leg, self.h_torso, self.w_torso), axis=-1)
            # except for joint no.21

    def load_data(self):
        # data: N C V T M
        # allow_pickle=True to support older inference NPZs that stored clip arrays as dtype=object
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.data_len = npz_data['len_train']
            if 'clips_train' in npz_data.files:
                self.sample_name = [str(x) for x in npz_data['clips_train'].tolist()]
            else:
                self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.data_len = npz_data['len_test']
            if 'clips_test' in npz_data.files:
                self.sample_name = [str(x) for x in npz_data['clips_test'].tolist()]
            else:
                self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        elif self.split == 'val':
            self.data = npz_data['x_val']
            self.label = np.where(npz_data['y_val'] > 0)[1]
            self.data_len = npz_data['len_val']
            if 'clips_val' in npz_data.files:
                self.sample_name = [str(x) for x in npz_data['clips_val'].tolist()]
            else:
                self.sample_name = ['val_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        data_len = self.data_len[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # modality
        if self.data_type == 'b':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
        elif self.data_type == 'jm':
            data_numpy = tools.to_motion(data_numpy)
        elif self.data_type == 'bm':
            j2b = tools.joint2bone()
            data_numpy = j2b(data_numpy)
            data_numpy = tools.to_motion(data_numpy)
        else:
            data_numpy = data_numpy.copy()

        if self.partition:
            data_numpy = data_numpy[:, :, self.new_idx]

        clip_name = self.sample_name[index]
        return data_numpy, data_len, label, index, clip_name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
