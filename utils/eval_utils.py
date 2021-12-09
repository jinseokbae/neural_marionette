import numpy as np
import torch

def evaluate(name, scores_dict, params):
    if name == 'semantic':
        return semantic_scores(scores_dict[name], params)
    elif name == 'voxel_chamfer':
        return voxel_chamfer_distance(scores_dict[name], params)
    else:
        raise ValueError("invalid evaluation metric.")

def evaluate_final(name, scores_dict):
    if name == 'semantic':
        scores = scores_dict[name]
        total_num = scores[0].sum()
        scores /= total_num
        scores = scores.max(axis=-1)  # (K',)
        np.savetxt('pretrained/results/semantic/semantic_result.csv', scores, delimiter=",")
        scores = scores.mean()
        return scores
    elif name == 'voxel_chamfer':
        scores = scores_dict[name]
        scores = np.array(scores)  # (totB, )
        np.savetxt('pretrained/results/chamfer/chamfer_result.csv', scores, delimiter=",")
        return scores.mean() * 1e4  # note that result is 1e4X
    else:
        raise ValueError("invalid evaluation metric.")

def voxel_chamfer_distance(scores, params):
    B, T, C, *X = params['voxel'].size()
    if scores is None:
        scores = []
    gt_voxel = params['voxel']
    recon = params['recon']
    gt_voxel = gt_voxel.squeeze(2)
    recon = recon.squeeze(2)
    recon[recon >= 0.5] = 1
    recon[recon < 0.5] = 0
    chamfer_tot_log = 0
    for b in range(B):
        chamfer_tot = 0
        for t in range(T):
            gt_coords = torch.stack(torch.where(gt_voxel[b, t]), dim=-1) / ((X[0] - 1) / 2) - 1  # (N, 3)
            recon_coords = torch.stack(torch.where(recon[b, t]), dim=-1) / ((X[0] - 1) / 2) - 1  # (M, 3)
            dist = (gt_coords[:, None] - recon_coords[None]).pow(2).sum(dim=-1)  # (N, M)
            chamfer = dist.min(dim=-1).values.mean() + dist.min(dim=0).values.mean()
            chamfer_tot += chamfer.item()
            chamfer_tot_log += chamfer.item()
        scores.append([chamfer_tot / T])

    log = dict(
        scores=scores,
        scores_log=chamfer_tot_log / (B * T)
    )
    return log



def semantic_scores(scores, params):
    B, T, K, _ = params['keypoints'].size()
    one_hot = np.array([np.array([0, ] * i + [1, ] + [0, ] * (K - 1 - i), dtype=np.int32) for i in range(K)])


    kypt = params['keypoints']
    gt_kypt = params['gt_keypoints']
    invalids = torch.where(kypt[..., -1] < 0.2)  # (B, T, K)
    kypt[invalids] = torch.Tensor([1e4, 1e4, 1e4, 1]).to(kypt.device)
    detected_kypt = kypt[:, :, :, :-1]  # (B, T, K, 3) = ~ (B, T, K, 4)
    detected_kypt = detected_kypt[:, :, None]  # (B, T, 1, K, 3)
    B, T, K_gt, _ = gt_kypt.size()  # (B, T, K', 3)

    if scores is None:
        scores = np.zeros((K_gt, K))
    gt_kypt = gt_kypt[:, :, :, None]  # (B, T, K', 1, 3)

    dist = (gt_kypt - detected_kypt).pow(2).sum(-1)  # (B, T, K', K)
    closest_kypt_idx = dist.min(dim=-1).indices  # (B, T, K')
    closest_kypt_idx = closest_kypt_idx.view(B * T, -1).detach().cpu().numpy()  # (B * T, K')
    temp = []
    for k_dot in range(K_gt):
        temp_k_dot = one_hot[closest_kypt_idx[:, k_dot]].sum(0)  # (B * T, K) --> (K, )
        scores[k_dot] += temp_k_dot
        temp.append((temp_k_dot / temp_k_dot.sum()).max())
    temp = np.array(temp, dtype=np.float32)

    log = dict(
        scores=scores,
        scores_log=temp.mean()
    )
    return log