import torch
import torch.nn as nn

def add_coord_channels(vox):

    """
    Adds channels containing pixel indices (x and y coordinates) to an image.
    Note: This has nothing to do with keypoint coordinates. It is just a data
    augmentation to allow convolutional networks to learn non-translation-
    equivariant outputs. This is similar to the "CoordConv" layers:
    https://arxiv.org/abs/1603.09382.
    Args:
    vox: [B, C, X1, X2, ..., XD] for dim D
    Returns:
    [B, C + D, X1, X2, ..., XD] tensor with x and y coordinate channels.
    """

    B, C, *X = vox.size()

    xd_grids = []
    for d in range(len(X)):
        xd_grid = torch.linspace(-1.0, 1.0, X[d], device=vox.device)  # [Xd, ]
        xd_grids.append(xd_grid)
    xd_maps = torch.stack(torch.meshgrid(*xd_grids), dim=0)  # [D, X1, X2, ..., XD]
    xd_maps = torch.stack([xd_maps] * B, dim=0)
    return torch.cat([vox, xd_maps], dim=1)  # concat along channel dim [B, C + D, X1, X2, ..., XD]

def extract_keypoints_from_heatmap(heatmap):
    # heatmap : (B, K, G1, G2, ..., GD)
    B, K, *G = heatmap.size()

    intensity_mean_dim = [i for i in range(2, len(G) + 2)]
    intensity = torch.mean(heatmap, dim=intensity_mean_dim)  # (B, K)
    intensity /= torch.max(intensity, dim=-1, keepdim=True).values + 1e-6  # (B, K)
    coords = []
    for d in range(len(G)):
        gd_grid = torch.linspace(-1.0, 1.0, G[d], device=heatmap.device)  # [Gd, ]
        reshape_size = [1] * (len(G) + 2)
        reshape_size[d + 2] = -1
        gd_grid = gd_grid.reshape(reshape_size) # (1, 1, 1, .. Gd, ..., 1)
        sum_dims = [i for i in range(2, d + 2)] + [i for i in range(d + 3, len(G) + 2)]
        weights = torch.sum(heatmap + 1e-6, dim=sum_dims, keepdim=True)  # (B, K, 1, ..., Gd, ..., 1)
        weights /= torch.sum(weights, dim=d + 2, keepdim=True)  # (B, K, 1, ..., Gd, ..., 1)
        coord_d = torch.sum(weights * gd_grid, dim=d + 2, keepdim=True)  # (B, K, 1, ..., 1)
        coord_d = torch.squeeze(coord_d)  # (B, K)
        if B == 1:
            coord_d = coord_d[None]
        if K == 1:
            coord_d = coord_d[:, None]
        coords.append(coord_d)
    coords = torch.stack(coords, dim=-1)  # (B, K, D)

    coords = torch.cat([coords, intensity[..., None]], dim=-1)  # (B, K, D + 1)

    return coords

def extract_gaussian_map_from_keypoints(keypoint, sigma=1.0, G=None):
    '''
    keypoint : (B, K, D + 1)
    '''

    coords = keypoint[..., :-1]  # (B, K, D)
    intensities = keypoint[..., -1]  # (B, K)


    B, K, D = coords.size()
    if type(sigma) is not list:
        gaussian_width = 2.0 * (sigma / G) ** 2.0
    else:
        gaussian_width = 2.0 * (torch.Tensor(sigma).to(keypoint.device) / G) ** 2.0
        gaussian_width = gaussian_width[None, :, None].expand(B, -1, G)


    map_dim = [G] * D
    map = torch.ones(B, K, *map_dim).to(keypoint.device)
    for d in range(D):
        xd_grid = torch.linspace(-1.0, 1.0, G, device=keypoint.device) # (output_map_width, )
        xd_grid = xd_grid[None, None].expand(B, K, -1)  # (B, K, output_map_width)
        xd_map = (-(xd_grid - coords[:, :, d][..., None]).pow(2) / gaussian_width).exp() # (B, K, output_map_width)
        for _ in range(d):
            xd_map = xd_map.unsqueeze(2)  # (B, K, 1, .., output_map_width)
        for _ in range(d + 1, D):
            xd_map = xd_map.unsqueeze(-1)  # (B, K, 1, .., output_map_width, ..., 1)
        map *= xd_map

    for _ in range(D):
        intensities = intensities.unsqueeze(-1)
    map = map * intensities

    return map

def get_keypoint_sparsity_loss(weight_matrix):
    """
    L1-loss on mean heatmap activations, to encourage sparsity.
    args:
      weight_matrix: heatmap output of shape [B, nkeypoints, G1, G2, ..., GD]
      factor -> weight
    """
    B, T, K, *G = weight_matrix.size()
    mean_dims = tuple([i for i in range(3, len(G) + 3)])
    heatmap_mean = torch.mean(weight_matrix, dim=mean_dims)  # (B, T, nkeypoints)
    sparsity_loss = torch.mean(torch.abs(heatmap_mean), dim=2)  # (B, T) sum along K
    return sparsity_loss

def get_temporal_separation_loss(keypoints, sep_sigma):
    '''
    :param keypoints: (B, T, K, (D + 1))
    :return:
    '''
    coords = keypoints[:, :, :, :-1]  # remove intensity
    B, T, K, D = coords.size()

    # Difference of keypoint coord of each time step with mean coord along T axis

    displacement = coords - coords.mean(dim=1, keepdim=True)  #(B, T, K, D)

    # Difference matrix of (B, T, K, K) (difference between displacements of each keypoint)
    difference = (displacement[:, :, :, None] - displacement[:, :, None]).pow(2).sum(-1)  # (B, T, K, K)

    # Temporal mean:
    difference = torch.mean(difference, dim=1)  # [B, K, K]

    # apply gaussian function to give lower loss for larger distance
    loss_matrix = (-difference / (2.0 * sep_sigma ** 2.0)).exp()  # (B, K, K)
    loss = loss_matrix.sum(dim=(1, 2))  # (B, )

    # Subtract sum of values on diagonal, which ar always 1 since exp(0) = 1
    loss = loss - K

    # normalize -> scale between 0 ~ 1
    loss = loss / (K * (K - 1))

    return loss  # (B)

def get_volume_fitting_loss(seq, keypoints, sigmas, vol_fit_type):
    B, T, C, *X = seq.size()
    _, _, K, *_ = keypoints.size()
    if vol_fit_type == 'none':
        vol_fit_reg = torch.zeros(B, T).to(seq.device)
    elif vol_fit_type == 'chamfer':
        vol_fit_reg = []
        for t in range(T):
            obs = seq[:, t]  # (B, 1, X, X, X)
            keypoint = keypoints[:, t, :, :3]  # (B, K, 3)
            obs = add_coord_channels(obs)
            obs = obs[:, None]  # (B, 1, 4, X, X, X)
            keypoint = keypoint[:, :, :, None, None, None]  # (B, K, 3, 1, 1, 1)
            dist = (obs[:, :, 1:] - keypoint).pow(2)  # (B, K, 3, X, X, X)
            dist = dist.sum(dim=2)  # (B, K, X, X, X)
            dist = dist.min(dim=1, keepdim=True).values  # (B, 1, X, X, X)
            dist = dist * seq[:, t]
            vol_fit_reg.append(dist.sum(dim=(1, 2, 3, 4)) / seq[:, t].sum(dim=(1, 2, 3, 4)))
        vol_fit_reg = torch.stack(vol_fit_reg, dim=1)
    elif vol_fit_type == 'gaussian':
        vol_fit_reg = []
        for t in range(T):
            keypoint = keypoints[:, t, :, :3]
            gaussian_mask = []
            for k in range(K):
                gaussian_mask.append(
                    extract_gaussian_map_from_keypoints(keypoint[:, k].unsqueeze(1), sigma=sigmas[k] * 4.0,
                                                        G=X[0]))
            gaussian_mask = torch.cat(gaussian_mask, dim=1).max(dim=1, keepdim=True).values  # (B, K, X, X, X)
            vol_fit_reg_t = (1 - gaussian_mask) * seq[:, t]  # (B, 1, X, X, X)
            vol_fit_reg_t = vol_fit_reg_t.sum(dim=(1, 2, 3, 4)) / seq[:, t].sum(dim=(1, 2, 3, 4)) # (B,
            vol_fit_reg.append(vol_fit_reg_t)
        vol_fit_reg = torch.stack(vol_fit_reg, dim=1)

    return vol_fit_reg


def get_graph_consistency_loss(keypoints, affinity, local_const=True, time_const=True, sparsity_const=True, intensity_const=True, ver=0):
    '''
    :param keypoints: B, T, K, D + 1
    :param affinity: nneighbor, K, K, 1
    :return:
    '''
    B, T, K, _ = keypoints.size()
    if local_const or time_const or intensity_const:
        influence = affinity.max(dim=0).values  # (K, K, 1)
        if ver == 2:
            influence = influence + influence.transpose(0, 1)
        positions = keypoints[:, :, :, :3]
        influence = influence[None, None]  # (1, 1, K, K, 1)
        intensities = keypoints[:, :, :, -1][..., None, None] # (B, T, K, 1, 1)

        dist = (positions[:, :, :, None] - positions[:, :, None]).pow(2).sum(dim=-1, keepdim=True)  # (B, T, K, K, 1)

    # loss for pulling nodes whose affinity score is high
    if local_const:
        if ver == 0 or ver == 2:
            local_consistency_loss = dist * influence * intensities  # (B, T, K, K, 1)
        elif ver == 1:
            local_consistency_loss = dist * influence  # (B, T, K, K, 1)
        local_consistency_loss = local_consistency_loss.mean(dim=(2, 3, 4))  # (B, T)
    else:
        local_consistency_loss = torch.zeros(1, 1).to(keypoints.device)

    # loss for time consistency
    if time_const:
        if ver == 0 or ver == 2:
            time_consistency_loss = abs(dist - dist.mean(dim=1, keepdim=True)) * influence * intensities
        elif ver == 1:
            time_consistency_loss = abs(dist - dist.mean(dim=1, keepdim=True)) * influence
        time_consistency_loss = time_consistency_loss.mean(dim=(2, 3, 4))  # (B, T)
    else:
        time_consistency_loss = torch.zeros(1, 1).to(keypoints.device)

    affinity = affinity.squeeze(-1)
    # each neighbor should be different
    if sparsity_const:
        affinity_self = affinity[:, None]  # (nneighbor, 1, K, K)
        affinity_other = affinity[None]  # (1, nneighbor, K, K)
        sparsity_loss = (affinity_self * affinity_other).pow(2).sum(dim=1, keepdim=True)  # (nneighbor, 1, K, K)
        sparsity_loss = sparsity_loss - affinity_self.pow(4)  # to remove self - self relation
        sparsity_loss = sparsity_loss.sum(dim=(0, 1))  # (K, K)
        sparsity_loss = sparsity_loss.mean(dim=(0, 1), keepdim=True)
    else:
        sparsity_loss = torch.zeros(1, 1).to(keypoints.device)


    intensity_loss = torch.zeros(1, 1).to(keypoints.device) # not used

    ## sparsity_loss is complex loss in the paper
    return local_consistency_loss, time_consistency_loss, sparsity_loss, intensity_loss


def get_graph_traj_loss(keypoints, affinity, ver=0):
    B, T, K, _ = keypoints.size()
    influence = affinity.squeeze(-1).max(dim=0).values
    if ver == 2:
        influence = influence + influence.transpose(0, 1)
    influence = influence[None, None]  # (1, 1, K, K)
    if ver == 0 or ver == 2:
        intensities = keypoints[:, :, :, -1].unsqueeze(-1)  # (B, T, K, 1)

    # # loss for enforcing similar trajectories for relationship with high affinity
    if ver == 0 or ver == 2:
        intensities_vel = (intensities[:, 1:] + intensities[:, :-1]) / 2  # (B, T - 1, K, 1)
    velocities = keypoints[:, 1:, :, :3] - keypoints[:, :-1, :, :3]  # (B, T - 1, K, 3)
    velocities_self = velocities[:, :, :, None]  # (B, T - 1, K, 1, 3)
    velocities_other = velocities[:, :, None]  # (B, T - 1, 1, K, 3)

    if ver == 0 or ver == 2:
        intensities_accel = (intensities_vel[:, 1:] + intensities_vel[:, :-1]) / 2
    accels = velocities[:, 1:] - velocities[:, :-1]  # (B, T - 2, K, 3)
    accels_self = accels[:, :, :, None]  # (B, T-2, K, 1, 3)
    accels_other = accels[:, :, None]  # (B, T-2, 1, K, 3)

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    # scaling (-1 ~ 1) --> (1 ~ 0)
    vel_cos = (-cos(velocities_self, velocities_other) + 1) / 2  # (B, T - 1, K, K)
    if ver == 0 or ver == 2:
        vel_cos = (vel_cos * influence * intensities_vel).mean(dim=(0, 1))
    elif ver == 1:
        vel_cos = (vel_cos * influence).mean(dim=(0, 1))
    accel_cos = (-cos(accels_self, accels_other) + 1) / 2  # (B, T - 2, K, K)
    if ver == 0 or ver == 2:
        accel_cos = (accel_cos * influence * intensities_accel).mean(dim=(0, 1))
    elif ver == 1:
        accel_cos = (accel_cos * influence).mean(dim=(0, 1))

    traj_consistency_loss = (vel_cos + accel_cos).mean(dim=(0, 1), keepdim=True)

    return traj_consistency_loss





