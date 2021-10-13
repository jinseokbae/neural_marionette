import os
import sys
import numpy as np


def crop_sequence(seq, start, T, sample_rate=1):
    return seq[start:start + T * sample_rate:sample_rate]

def episodic_normalization(seq, scale=1.0, x_trans=0.0, z_trans=0.0, joints=None):
    # seq = (T, N, 3)
    bmax = np.amax(seq, axis=(0, 1))
    bmin = np.amin(seq, axis=(0, 1))
    # bmin = bmin - abs(np.array([x_trans, 0, z_trans]))
    blen = (bmax - bmin).max()
    seq = ((seq - bmin[None, None]) * scale / (blen + 1e-5)) * 2 - 1 + np.array([x_trans, 0, z_trans])
    if joints is not None:
        joints = ((joints- bmin[None, None]) * scale / (blen + 1e-5)) * 2 - 1
        return seq, joints
    return seq

def voxelize(pos_coords, output_shape, is_binarized=True):
    if not is_binarized:
        vertex_normals = pos_coords[..., 3:]
    bbox = np.array([-1, -1, -1] + [1, 1, 1])
    step = (bbox[3:] - bbox[:3]) / output_shape
    occupancy_grid = np.zeros(output_shape, dtype=np.float32)
    pos_coords = pos_coords[..., :3]
    idxs = ((pos_coords - bbox[:3]) / (step + 1e-5)).astype(np.int32)
    occupancy_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1.0

    return occupancy_grid[None]  # (1, grid, grid, grid)




