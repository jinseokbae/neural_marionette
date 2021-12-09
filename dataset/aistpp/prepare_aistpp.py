import os
from smplx import SMPL
import trimesh
import matplotlib.pyplot as plt
from aist_plusplus.loader import AISTDataset
import torch
import celluloid
import numpy as np
from scipy.spatial.transform import Rotation as R
import random


def sample_faces(mesh, N=20000):
    P, t = trimesh.sample.sample_surface(mesh, N)
    sampled_faces = np.hstack([P, mesh.face_normals[t, :]]).astype(np.float32)
    return sampled_faces  # (N, 3 + 3)


if __name__ == "__main__":
    anno_dir = 'aist_plusplus_final'
    smpl_dir = 'smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl'
    save_dir = 'aist_plusplus_smpl_joints'
    random_seed = 0
    random.seed(0)
    seqs_ori = sorted(os.listdir(os.path.join(anno_dir, 'motions')))
    with open(os.path.join(anno_dir, 'ignore_list.txt'), 'rb') as f:
        ignores = f.read().splitlines()
        ignores = ignores[:-1]
        ignores = [a.decode("utf-8") + '.pkl' for a in ignores]
    seqs = []
    for seq in seqs_ori:
        flag = False
        for ignore in ignores:
            if seq[:26] == ignore[:26]:
                flag = True
                break
        if not flag:
            seqs.append(seq[:-4])
    print(seqs_ori.__len__())
    print(ignores.__len__())
    print(seqs.__len__())

    aist_dataset = AISTDataset(anno_dir)

    os.makedirs(os.path.join(save_dir, 'surface', 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'surface', 'test'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'root_aligns', 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'root_aligns', 'test'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'joints', 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'joints', 'test'), exist_ok=True)
    random.shuffle(seqs)
    tot_len = len(seqs)
    for idx, seq in enumerate(seqs):
        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
            aist_dataset.motion_dir, seq)
        smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)
        vertices = smpl.forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
            transl=torch.from_numpy(smpl_trans).float(),
            scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
        ).vertices.detach()
        faces = smpl.faces
        # affinity
        if idx == 0:
            affinity = torch.zeros(24, 24)
            for k in range(24):
                parent = smpl.parents[k]
                if parent < 0:
                    continue
                affinity[k, parent] = 1
            affinity = torch.max(affinity, affinity.transpose(0, 1)).detach().numpy()
            np.save(os.path.join(save_dir, 'gt_affinity.npy'), affinity)

        sampled = []
        root_aligns = []
        for t in range(vertices.shape[0]):
            mesh = trimesh.Trimesh(vertices[t], faces)
            sampled.append(sample_faces(mesh, N=20000))
            r = R.from_rotvec(smpl_poses[t, :3]).as_euler('xyz', degrees=True)
            ry = R.from_euler('y', r[1], degrees=True).as_matrix().T
            root_aligns.append(ry)
        sampled = np.stack(sampled, axis=0)[..., :3]
        root_aligns = np.stack(root_aligns, axis=0)

        # keypoints
        J_regressor = smpl.J_regressor
        J_regressor = J_regressor[None].expand(vertices.shape[0], -1, -1)
        J = torch.einsum('bij, bjk->bik', J_regressor, vertices).detach().numpy()  # (T, K, 3)

        if (idx / tot_len) <= 0.9:
            np.save(os.path.join(save_dir, 'surface', 'train', seq + '.npy'), sampled)
            np.save(os.path.join(save_dir, 'root_aligns', 'train', seq + '.npy'), root_aligns)
            np.save(os.path.join(save_dir, 'joints', 'train', seq + '.npy'), J)
            print('%d / %d' % (idx, tot_len), 'train', seq, 'saved.')
            with open('train_list.txt', 'a') as f:
                f.write(seq + "\n")
        else:
            np.save(os.path.join(save_dir, 'surface', 'test', seq + '.npy'), sampled)
            np.save(os.path.join(save_dir, 'root_aligns', 'test', seq + '.npy'), root_aligns)
            np.save(os.path.join(save_dir, 'joints', 'test', seq + '.npy'), J)
            print('%d / %d' % (idx, tot_len), 'test', seq, 'saved.')
            with open('test_list.txt', 'a') as f:
                f.write(seq + "\n")