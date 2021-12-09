import os
import torch
import torch.nn.functional as F

# Util function for loading point clouds|
import numpy as np
import matplotlib.pyplot as plt
from utils.vis_utils import Arrow3D

# Color map
import matplotlib.cm as cm
import cv2
plt.rcParams.update({'figure.max_open_warning': 0})

def vis_keypoints(vox, keypoints, logger_path, nepoch, affinity=None, log_num=8, group='track', mode='affinity', Tcond=3):
    '''
    :param keypoitns: (B, T, K, 4)
    :param affinity: (K, K, 1)
    :param logg_erpath:
    :param nepoch:
    :param log_num:
    :param group:
    :return:
    '''
    save_gif_root = os.path.join(logger_path, 'gifs', str(nepoch), group)
    os.makedirs(save_gif_root, exist_ok=True)

    B, T, K, D = keypoints.size()
    _, _, _, *X = vox.size()
    assert D == 4

    if log_num > B:
        log_num = B
    vox = vox[:log_num]
    keypoints = keypoints[:log_num]

    if affinity is not None:
        if mode == 'affinity':
            nneighbor, *_ = affinity.size()
            affinity = affinity.max(dim=0).values
        affinity = affinity.squeeze(-1).detach().cpu().numpy()  # (K, K)

    gif = []
    for b in range(log_num):
        gif_b = []
        for t in range(T):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_xlabel('X-axis', fontweight='bold')
            ax.set_ylabel('Z-axis', fontweight='bold')
            ax.set_zlabel('Y-axis', fontweight='bold')
            vox_bt = vox[b, t, 0].detach().cpu().numpy()
            coords = np.stack(np.where(vox_bt), axis=-1) / ((X[0] - 1) / 2) - 1
            kypts = keypoints[b, t, :, :3]  # (N, 3)
            kypts = kypts.detach().cpu().numpy()
            if group == 'gen' and t >= Tcond:
                color = [0, 0, 0.8]
            else:
                color = 'grey'
            ax.scatter3D(coords[:, 0], -coords[:, 2], coords[:, 1], color=color, s=2, alpha=0.3)
            alphas = keypoints[b, t, :, -1].detach().cpu().numpy().clip(0, 1)
            max_kypt_alpha = alphas.max() + 1e-5
            if mode == 'A' or mode == 'A_hats':
                matrix = affinity[b, t]
                matrix_max_alpha = affinity[b, t].max()
            for k in range(K):
                ax.plot(kypts[k, 0], -kypts[k, 2], kypts[k, 1], color='red', marker='o', markersize=6, linewidth=0, alpha=alphas[k] / max_kypt_alpha)
                if affinity is not None:
                    if mode == 'affinity':
                        temp_row = affinity[k].copy()
                        neighbor_idx = temp_row.argsort()[::-1][:nneighbor]
                        matrix = affinity
                        for kdot in neighbor_idx:
                            pairs = np.stack([kypts[k], kypts[kdot]], axis=0)
                            directed_edge = Arrow3D([pairs[0, 0], pairs[1, 0]], [-pairs[0, 2], -pairs[1, 2]],
                                                    [pairs[0, 1], pairs[1, 1]], mutation_scale=10,
                                                    lw=1.7, arrowstyle="-|>", color="g",
                                                    alpha=alphas[k] / max_kypt_alpha)
                            ax.add_artist(directed_edge)
                    elif mode == 'A' or mode == 'A_hats':
                        neighbor_idx = np.arange(k, K)
                        for kdot in neighbor_idx:
                            pairs = np.stack([kypts[k], kypts[kdot]], axis=0)
                            ax.plot(pairs[..., 0], -pairs[..., 2], pairs[..., 1], color='green', marker='o',
                                    markersize=0, linewidth=2.5, alpha=matrix[k, kdot].clip(0, 1))




            ax.view_init(elev=10, azim=-60)

            save_file = os.path.join(save_gif_root, 'keypoints_%d_%d.png' % (b, t))
            fig.savefig(save_file)
            plt.close(fig)
            image = cv2.imread(save_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gif_b.append(torch.from_numpy(image).to(keypoints.device))
        gif_b = torch.stack(gif_b, dim=0)
        gif.append(gif_b)
    gif = torch.stack(gif, dim=0)  # (B, T, H, W, C)

    return gif.permute(0, 1, 4, 2, 3)



def vis_recon(vox, recon, logger_path, nepoch, log_num=8, group='track', Tcond=3):
    '''
    :param vox: (B, T, 1, X, X, X)
    :param recon: (B, T, 1, X, X, X)
    :param log_num: logging num
    :param nepoch:
    :param divide: list of split
    :return: gif : torch.Tensor (B, T, C, H, W)
    '''
    save_gif_root = os.path.join(logger_path, 'gifs', str(nepoch), group)
    os.makedirs(save_gif_root, exist_ok=True)

    B, T, C, *X = vox.size()

    if log_num > B:
        log_num = B
    vox = vox[:log_num]
    recon = recon[:log_num]
    recon[recon < 0.5] = 0
    recon[recon >= 0.5] = 1
    gif = []
    for b in range(log_num):
        gif_b = []
        for t in range(T):
            coords_gt = torch.stack(torch.where(vox[b, t, 0]), dim=-1) / ((X[0] - 1) / 2) - 1  # (N, 3)
            coords_recon = torch.stack(torch.where(recon[b, t, 0]), dim=-1) / ((X[0] - 1) / 2) - 1  # (N, 3)
            coords_gt = coords_gt.detach().cpu().numpy()
            coords_recon = coords_recon.detach().cpu().numpy()

            gif_b_azim = []
            for azim in [-60]:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_zlim(-1, 2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_xlabel('X-axis', fontweight='bold')
                ax.set_ylabel('Z-axis', fontweight='bold')
                ax.set_zlabel('Y-axis', fontweight='bold')
                ax.scatter3D(coords_gt[..., 0] - 0.5, -coords_gt[..., 2], coords_gt[..., 1], color='grey', s=2)
                if group == 'gen' and t >= Tcond:
                    color = 'blue'
                else:
                    color = 'green'
                ax.scatter3D(coords_recon[..., 0] + 0.5, -coords_recon[..., 2], coords_recon[..., 1], color=color, s=2)
                ax.view_init(elev=10, azim=azim)
                save_file = os.path.join(save_gif_root, 'recon_%d_%d.png' % (b, t))
                plt.savefig(save_file)
                image = cv2.imread(save_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gif_b_azim.append(torch.from_numpy(image).to(vox.device))
            gif_b_azim = torch.cat(gif_b_azim, dim=1)
            gif_b.append(gif_b_azim)
        gif_b = torch.stack(gif_b, dim=0)
        gif.append(gif_b)
    gif = torch.stack(gif, dim=0)  # (B, T, H, W, C)

    return gif.permute(0, 1, 4, 2, 3)