from __future__  import print_function
import os
import json
import glob
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import pickle
from src.utils.dataset_utils import crop_sequence, episodic_normalization, voxelize

class DFAUST(data.Dataset):
    def __init__(self, train=True, options=None):
        split = 'train' if train else 'test'
        self.is_binarized = options.is_binarized
        # self.vox_root = os.path.join('data/D-FAUST/vox32_fpsx5', split)
        self.root = os.path.join('data/D-FAUST/surface', split)
        self.is_eval = bool(options.is_eval)
        self.T = options.Ttot
        self.sample_rate = options.sample_rate
        self.grid_size = options.grid_size
        self._output_shape = (self.grid_size,) * 3
        self.random_crop = bool(options.random_crop)

        sid_list = sorted(os.listdir(self.root))
        self.seq_path = []
        for sid in sid_list:
            seq_list = sorted(os.listdir(os.path.join(self.root, sid)))
            for seq in seq_list:
                self.seq_path.append(os.path.join(sid, seq))

        random.seed(options.seed)
        random.shuffle(self.seq_path)
        if options.debug == 1:
            self.seq_path = self.seq_path[:options.nbatch]


    def log_epoch(self, epoch_id):
        self.epoch_id = epoch_id

    def _now_epoch(self):
        # for debugging
        breakpoint()

    def __getitem__(self, index):

        x = np.load(os.path.join(self.root, self.seq_path[index]))[..., :3]

        if self.random_crop:
            rand_start = x.shape[0] - 1 - self.sample_rate * (self.T - 1)
            if rand_start < 0:
                start = 0
            else:
                start = random.randint(0, x.shape[0] - 1 - self.sample_rate * (self.T - 1))
        else:
            offset = (self.epoch_id % self.T) * self.sample_rate
            start = self.epoch_id % (x.shape[0] // (self.T * self.sample_rate)) * (self.T * self.sample_rate) + offset
            if start + (self.T - 1) * self.sample_rate >= x.shape[0]:
                # back = x.shape[0] - (start + (self.T - 1) * self.sample_rate)
                # start -= back
                start = max(start - 2 * offset, 0)

        if x.shape[0] < self.T * self.sample_rate:
            start = 0
            copy_num = self.T - x.shape[0]
            x = np.concatenate([x] + [x[-1:]] * copy_num, axis=0)

        x = crop_sequence(x,
                        start=start,
                        T=self.T,
                        sample_rate=self.sample_rate)
        x = episodic_normalization(x)

        T, N, _ = x.shape


        vox_seq = []
        for t in range(T):
            # idx = np.random.choice(N, self.N)
            try:
                vox_seq.append(voxelize(x[t], self._output_shape, self.is_binarized))
            except:
                raise ValueError("Dataset voxelizer error")
        vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float()

        return vox_seq

    def __len__(self):
        return len(self.seq_path)


class AIST(data.Dataset):
    def __init__(self, train=True, options=None, align_root=False):
        self.is_eval = bool(options.is_eval)
        split = 'train' if train else 'test'
        self.align_root = align_root
        self.is_eval = bool(options.is_eval)
        self.is_binarized = options.is_binarized
        # self.vox_root = os.path.join('data/D-FAUST/vox32_fpsx5', split)
        self.root = os.path.join('data/aist_plusplus_smpl_joints/surface', split)
        self.root_aligns = os.path.join('data/aist_plusplus_smpl_joints/root_aligns', split)
        self.joint_root = os.path.join('data/aist_plusplus_smpl_joints/joints', split)
        self.T = options.Ttot
        self.sample_rate = options.sample_rate
        self.grid_size = options.grid_size
        self._output_shape = (self.grid_size,) * 3
        self.random_crop = bool(options.random_crop)

        self.seq_path = sorted(os.listdir(self.root))
        if self.align_root:
            self.r_path = sorted(os.listdir(self.root_aligns))

        random.seed(options.seed)
        random.shuffle(self.seq_path)
        if options.debug == 1:
            self.seq_path = self.seq_path[:options.nbatch]

    def log_epoch(self, epoch_id):
        self.epoch_id = epoch_id

    def __getitem__(self, index):

        x = np.load(os.path.join(self.root, self.seq_path[index]))[..., :3]
        if self.is_eval:
            joints = np.load(os.path.join(self.joint_root, self.seq_path[index]))

        if self.random_crop:
            rand_start = x.shape[0] - 1 - self.sample_rate * (self.T - 1)
            if rand_start < 0:
                start = 0
            else:
                start = random.randint(0, x.shape[0] - 1 - self.sample_rate * (self.T - 1))
        else:
            offset = (self.epoch_id % self.T) * self.sample_rate
            start = self.epoch_id % (x.shape[0] // (self.T * self.sample_rate)) * (self.T * self.sample_rate) + offset
            if start + (self.T - 1) * self.sample_rate >= x.shape[0]:
                # back = x.shape[0] - (start + (self.T - 1) * self.sample_rate)
                # start -= back
                start = max(start - 2 * offset, 0)

        if x.shape[0] < self.T * self.sample_rate:
            start = 0
            copy_num = self.T - x.shape[0]
            x = np.concatenate([x] + [x[-1:]] * copy_num, axis=0)

        x = crop_sequence(x,
                        start=start,
                        T=self.T,
                        sample_rate=self.sample_rate)  # [self.T, N, 3]
        if self.is_eval:
            joints = crop_sequence(joints,
                                   start=start,
                                   T=self.T,
                                   sample_rate=self.sample_rate)
            x, joints = episodic_normalization(x, joints=joints)
        else:
            x = episodic_normalization(x)

        if self.align_root:
            r = np.load(os.path.join(self.root_aligns, self.seq_path[index]))  #[T, 3, 3]
            r = r[start][None].expand(self.T, -1, -1)  # [self.T, 3, 3]
            x = np.einsum('tij, tjk', r, x.transpose(2, 1)).transpose(2, 1)  #[self.T, N, 3]



        T, N, _ = x.shape


        vox_seq = []
        for t in range(T):
            # idx = np.random.choice(N, self.N)
            try:
                vox_seq.append(voxelize(x[t], self._output_shape, self.is_binarized))
            except:
                raise ValueError("Dataset voxelizer error")
        vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float()

        if self.is_eval:
            return vox_seq, joints
        else:
            return vox_seq

    def __len__(self):
        return len(self.seq_path)

class DeformingThings4DAnimals(data.Dataset):
    def __init__(self, train=True, options=None):

        self.is_eval = bool(options.is_eval)
        split = 'train' if train else 'test'
        self.is_binarized = options.is_binarized
        # self.vox_root = os.path.join('data/D-FAUST/vox32_fpsx5', split)
        self.root = os.path.join('data/DeformingThings4D/animals', split)
        self.T = options.Ttot
        self.sample_rate = options.sample_rate
        self.grid_size = options.grid_size
        self._output_shape = (self.grid_size,) * 3
        self.random_crop = bool(options.random_crop)

        sid_list = sorted(os.listdir(self.root))
        self.seq_path = []
        for sid in sid_list:
            seq_list = sorted(os.listdir(os.path.join(self.root, sid)))
            for seq in seq_list:
                self.seq_path.append(os.path.join(sid, seq))

        random.seed(options.seed)
        random.shuffle(self.seq_path)
        if options.debug == 1:
            self.seq_path = self.seq_path[:options.nbatch]


    def log_epoch(self, epoch_id):
        self.epoch_id = epoch_id

    def __getitem__(self, index):

        x = np.load(os.path.join(self.root, self.seq_path[index]))[..., :3]
        if x.shape[0] < self.T * self.sample_rate:
            start = 0
            copy_num = self.T - x.shape[0]
            x = np.concatenate([x] + [x[-1:]] * copy_num, axis=0)
        else:
            if self.random_crop:
                rand_start = x.shape[0] - 1 - self.sample_rate * (self.T - 1)
                if rand_start < 0:
                    start = 0
                else:
                    start = random.randint(0, x.shape[0] - 1 - self.sample_rate * (self.T - 1))
            else:
                offset = (self.epoch_id % self.T) * self.sample_rate
                start = self.epoch_id % (x.shape[0] // (self.T * self.sample_rate)) * (self.T * self.sample_rate) + offset
                if start + (self.T - 1) * self.sample_rate >= x.shape[0]:
                    # back = x.shape[0] - (start + (self.T - 1) * self.sample_rate)
                    # start -= back
                    start = max(start - 2 * offset, 0)

        x = crop_sequence(x,
                        start=start,
                        T=self.T,
                        sample_rate=self.sample_rate)
        x = episodic_normalization(x)

        T, N, _ = x.shape


        vox_seq = []
        for t in range(T):
            # idx = np.random.choice(N, self.N)
            try:
                vox_seq.append(voxelize(x[t], self._output_shape, self.is_binarized))
            except:
                raise ValueError("Dataset voxelizer error")
        vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float()

        return vox_seq

    def __len__(self):
        return len(self.seq_path)

class DeformingThings4DHumanoids(data.Dataset):
    def __init__(self, train=True, options=None):

        self.is_eval = bool(options.is_eval)
        split = 'train' if train else 'test'
        self.is_binarized = options.is_binarized
        # self.vox_root = os.path.join('data/D-FAUST/vox32_fpsx5', split)
        self.root = os.path.join('data/DeformingThings4D/humanoids', split)
        self.T = options.Ttot
        self.sample_rate = options.sample_rate
        self.grid_size = options.grid_size
        self._output_shape = (self.grid_size,) * 3
        self.random_crop = bool(options.random_crop)
        sid_list = sorted(os.listdir(self.root))
        self.seq_path = []
        for sid in sid_list:
            seq_list = sorted(os.listdir(os.path.join(self.root, sid)))
            for seq in seq_list:
                self.seq_path.append(os.path.join(sid, seq))

        random.seed(options.seed)
        random.shuffle(self.seq_path)
        if options.debug == 1:
            self.seq_path = self.seq_path[:options.nbatch]


    def log_epoch(self, epoch_id):
        self.epoch_id = epoch_id

    def __getitem__(self, index):

        x = np.load(os.path.join(self.root, self.seq_path[index]))[..., :3]
        if x.shape[0] < self.T * self.sample_rate:
            start = 0
            copy_num = self.T - x.shape[0]
            x = np.concatenate([x] + [x[-1:]] * copy_num, axis=0)
        else:
            if self.random_crop:
                rand_start = x.shape[0] - 1 - self.sample_rate * (self.T - 1)
                if rand_start < 0:
                    start = 0
                else:
                    start = random.randint(0, x.shape[0] - 1 - self.sample_rate * (self.T - 1))
            else:
                offset = (self.epoch_id % self.T) * self.sample_rate
                start = self.epoch_id % (x.shape[0] // (self.T * self.sample_rate)) * (self.T * self.sample_rate) + offset
                if start + (self.T - 1) * self.sample_rate >= x.shape[0]:
                    # back = x.shape[0] - (start + (self.T - 1) * self.sample_rate)
                    # start -= back
                    start = max(start - 2 * offset, 0)

        x = crop_sequence(x,
                        start=start,
                        T=self.T,
                        sample_rate=self.sample_rate)
        x = episodic_normalization(x)

        T, N, _ = x.shape


        vox_seq = []
        for t in range(T):
            # idx = np.random.choice(N, self.N)
            try:
                vox_seq.append(voxelize(x[t], self._output_shape, self.is_binarized))
            except:
                raise ValueError("Dataset voxelizer error")
        vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float()

        return vox_seq

    def __len__(self):
        return len(self.seq_path)

class Panda(data.Dataset):
    def __init__(self, train=True, options=None):

        split = 'train' if train else 'test'
        self.is_binarized = options.is_binarized
        # self.vox_root = os.path.join('data/D-FAUST/vox32_fpsx5', split)
        self.root = os.path.join('data/panda_gripper', split, 'vertices')
        self.joint_root = os.path.join('data/panda_gripper', split, 'centroids')
        self.T = options.Ttot
        self.sample_rate = options.sample_rate
        self.grid_size = options.grid_size
        self._output_shape = (self.grid_size,) * 3
        self.is_eval = bool(options.is_eval)
        self.random_crop = bool(options.random_crop)
        self.seq_path = sorted(os.listdir(self.root))

        random.seed(options.seed)
        random.shuffle(self.seq_path)
        if options.debug == 1:
            self.seq_path = self.seq_path[:options.nbatch]

    def log_epoch(self, epoch_id):
        self.epoch_id = epoch_id

    def __getitem__(self, index):
        vertices_file = os.path.join(self.root, self.seq_path[index])
        joints_file = os.path.join(self.joint_root, self.seq_path[index])
        joints_file = joints_file.split('_')[0] + '_' + joints_file.split('_')[1] + '_centroids.npy'
        x = np.load(vertices_file)[..., :3]
        if self.is_eval:
            joints = np.load(joints_file)  # (T, K', 3)
        if x.shape[0] < self.T * self.sample_rate:
            start = 0
            copy_num = self.T - x.shape[0]
            x = np.concatenate([x] + [x[-1:]] * copy_num, axis=0)
        else:
            if self.random_crop:
                rand_start = x.shape[0] - 1 - self.sample_rate * (self.T - 1)
                if rand_start < 0:
                    start = 0
                else:
                    start = random.randint(0, x.shape[0] - 1 - self.sample_rate * (self.T - 1))
            else:
                offset = (self.epoch_id % self.T) * self.sample_rate
                start = self.epoch_id % (x.shape[0] // (self.T * self.sample_rate)) * (self.T * self.sample_rate) + offset
                if start + (self.T - 1) * self.sample_rate >= x.shape[0]:
                    # back = x.shape[0] - (start + (self.T - 1) * self.sample_rate)
                    # start -= back
                    start = max(start - 2 * offset, 0)

        x = crop_sequence(x,
                        start=start,
                        T=self.T,
                        sample_rate=self.sample_rate)
        if self.is_eval:
            joints = crop_sequence(joints,
                                   start=start,
                                   T=self.T,
                                   sample_rate=self.sample_rate)
            x, joints = episodic_normalization(x, joints=joints)
        else:
            x = episodic_normalization(x)
        T, N, _ = x.shape
        vox_seq = []
        for t in range(T):
            # idx = np.random.choice(N, self.N)
            try:
                vox_seq.append(voxelize(x[t], self._output_shape, self.is_binarized))
            except:
                raise ValueError("Dataset voxelizer error")
        vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float()
        if self.is_eval:
            return vox_seq, joints
        else:
            return vox_seq

    def __len__(self):
        return len(self.seq_path)

class InterHand(data.Dataset):
    def __init__(self, train=True, options=None, N=4096):
        split = 'train' if train else 'test'
        self.is_binarized = options.is_binarized
        # self.vox_root = os.path.join('data/D-FAUST/vox32_fpsx5', split)
        self.root = os.path.join('data/InterHand2.6Mnpy', split)
        self.T = options.Ttot
        self.sample_rate = options.sample_rate
        self.grid_size = options.grid_size
        self._output_shape = (self.grid_size,) * 3
        self.random_crop = bool(options.random_crop)
        self.seq_path = []
        self.scale = 0.7
        episodes = sorted(os.listdir(self.root))
        for episode in episodes:
            hand_types = sorted(os.listdir(os.path.join(self.root, episode)))
            for hand_type in hand_types:
                files = sorted(os.listdir(os.path.join(self.root, episode, hand_type)))
                for file in files:
                    self.seq_path.append(os.path.join(episode, hand_type, file))
        random.seed(options.seed)
        random.shuffle(self.seq_path)
        if options.debug == 1:
            self.seq_path = self.seq_path[:options.nbatch]
    def log_epoch(self, epoch_id):
        self.epoch_id = epoch_id
    def __getitem__(self, index):
        x = np.load(os.path.join(self.root, self.seq_path[index]))[..., :3]
        if self.random_crop:
            rand_start = x.shape[0] - 1 - self.sample_rate * (self.T - 1)
            if rand_start < 0:
                start = 0
            else:
                start = random.randint(0, x.shape[0] - 1 - self.sample_rate * (self.T - 1))
        else:
            offset = (self.epoch_id % self.T) * self.sample_rate
            start = self.epoch_id % (x.shape[0] // (self.T * self.sample_rate)) * (self.T * self.sample_rate) + offset
            if start + (self.T - 1) * self.sample_rate >= x.shape[0]:
                # back = x.shape[0] - (start + (self.T - 1) * self.sample_rate)
                # start -= back
                start = max(start - 2 * offset, 0)
        if x.shape[0] < self.T * self.sample_rate:
            start = 0
            copy_num = self.T - x.shape[0]
            x = np.concatenate([x] + [x[-1:]] * copy_num, axis=0)
        x = crop_sequence(x,
                        start=start,
                        T=self.T,
                        sample_rate=self.sample_rate)  # [self.T, N, 3]
        x = episodic_normalization(x, self.scale)
        T, N, _ = x.shape
        vox_seq = []
        for t in range(T):
            # idx = np.random.choice(N, self.N)
            try:
                vox_seq.append(voxelize(x[t], self._output_shape, self.is_binarized))
            except:
                raise ValueError("Dataset voxelizer error")
        vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float()
        return vox_seq
    def __len__(self):
        return len(self.seq_path)

class HanCo(data.Dataset):
    def __init__(self, train=True, options=None):

        split = 'train' if train else 'test'
        self.is_binarized = options.is_binarized
        self.root = os.path.join('data/HanCo', split, 'vertices')
        self.joint_root = os.path.join('data/HanCo', split, 'joints')
        self.T = options.Ttot
        self.sample_rate = options.sample_rate
        self.grid_size = options.grid_size
        self._output_shape = (self.grid_size,) * 3
        self.random_crop = bool(options.random_crop)
        self.is_eval = bool(options.is_eval)

        self.seq_path = sorted(os.listdir(self.root))

        random.seed(options.seed)
        random.shuffle(self.seq_path)
        if options.debug == 1:
            self.seq_path = self.seq_path[:options.nbatch]

    def log_epoch(self, epoch_id):
        self.epoch_id = epoch_id

    def __getitem__(self, index):
        vertices_file = os.path.join(self.root, self.seq_path[index])
        joints_file = os.path.join(self.joint_root, self.seq_path[index].split('_')[0] + '_joints.npy')

        x = np.load(vertices_file)[..., :3]
        if self.is_eval:
            joints = np.load(joints_file)  # (T, K', 3)

        if self.random_crop:
            rand_start = x.shape[0] - 1 - self.sample_rate * (self.T - 1)
            if rand_start < 0:
                start = 0
            else:
                start = random.randint(0, x.shape[0] - 1 - self.sample_rate * (self.T - 1))
        else:
            offset = (self.epoch_id % self.T) * self.sample_rate
            start = self.epoch_id % (x.shape[0] // (self.T * self.sample_rate)) * (self.T * self.sample_rate) + offset
            if start + (self.T - 1) * self.sample_rate >= x.shape[0]:
                # back = x.shape[0] - (start + (self.T - 1) * self.sample_rate)
                # start -= back
                start = max(start - 2 * offset, 0)

        if x.shape[0] < self.T * self.sample_rate:
            start = 0
            copy_num = self.T - x.shape[0]
            x = np.concatenate([x] + [x[-1:]] * copy_num, axis=0)

        x = crop_sequence(x,
                        start=start,
                        T=self.T,
                        sample_rate=self.sample_rate)  # [self.T, N, 3]

        if self.is_eval:
            joints = crop_sequence(joints,
                                   start=start,
                                   T=self.T,
                                   sample_rate=self.sample_rate)
            x, joints = episodic_normalization(x, joints=joints)

        else:
            x = episodic_normalization(x)

        T, N, _ = x.shape


        vox_seq = []
        for t in range(T):
            # idx = np.random.choice(N, self.N)
            try:
                vox_seq.append(voxelize(x[t], self._output_shape, self.is_binarized))
            except:
                raise ValueError("Dataset voxelizer error")
        vox_seq = torch.from_numpy(np.stack(vox_seq, axis=0)).float()

        if self.is_eval:
            return vox_seq, joints
        else:
            return vox_seq

    def __len__(self):
        return len(self.seq_path)

class DATASET_LIST:
    """list of all the dataset"""
    def __init__(self):

        self.datasets = {
            "dfaust": DFAUST,
            "aist": AIST,
            "animals" : DeformingThings4DAnimals,
            "humanoids" : DeformingThings4DHumanoids,
            "panda" : Panda,
            "hands" : InterHand,
            "hanco" : HanCo
        }

        self.type = self.datasets.keys()

    def load(self,training,options):

        if training:
            print("\nTRAINING DATASET:")
        else:
            print("VALIDATION DATASET:")
        dataset = self.datasets[options.dataset](training, options)
        print("\n")
        return dataset
