import torch
from torch import nn
import torch.nn.functional as F
from utils.kypt_detector_utils import add_coord_channels, get_keypoint_sparsity_loss, \
    get_temporal_separation_loss, get_volume_fitting_loss, extract_keypoints_from_heatmap, \
    extract_gaussian_map_from_keypoints, get_graph_consistency_loss, get_graph_traj_loss
from modules.vox_modules import HG, Basic3DBlock, Res3DBlock, Pool3DBlock


class KyptDetector(nn.Module):
    """
    Kypt detector for voxel data
    """
    def __init__(self, options):
        nn.Module.__init__(self)

        # experimental
        self.vol_fit_type = options.vol_fit_type
        self.fixed_sigma = bool(options.fixed_sigma)
        self.keypoints_graph = options.keypoints_graph
        self.keypoints_detach = bool(options.keypoints_detach)
        self.graph_random_init = bool(options.graph_random_init)
        self.using_local_const = bool(options.using_local_const)
        self.using_time_const = bool(options.using_time_const)
        self.using_sparsity_const = bool(options.using_sparsity_const)
        self.using_intensity_const = bool(options.using_intensity_const)
        self.using_graph_traj = options.graph_traj_weight > 0
        self.using_graph_vol = options.graph_vol_weight > 0
        self.affinity_ver = options.affinity_ver
        self.graph_loss_ver = options.graph_loss_ver
        self.gaussian_sigma = options.gaussian_sigma

        # config
        self.is_binarized = options.is_binarized
        self.input_dim = options.input_dim
        self.grid_size = options.grid_size

        self.nkeypoints = options.nkeypoints

        sigmas = [self.gaussian_sigma] * self.nkeypoints
        self.sigmas = sigmas

        self.vox_to_kypt = VoxToKyptNet(grid_size=options.grid_size, nkeypoints=options.nkeypoints,
                                        input_dim=options.input_dim, sigmas=sigmas,
                                        fixed_sigma=bool(options.fixed_sigma), const_intensity=options.const_intensity)
        self.kypt_to_vox = KyptToVoxNet(grid_size=options.grid_size, nkeypoints=options.nkeypoints, input_dim=options.input_dim, gaussian_cat_type=options.gaussian_cat_type)

        self.sep_sigma = options.sep_sigma

        self.affinity_anneal = options.affinity_anneal
        self.affinity_start = False

        # graph related
        if self.keypoints_graph == 'affinity_params':
            self.nneighbor = options.nneighbor
            if self.graph_random_init:
                if self.affinity_ver < 3:
                    self.affinity_params = nn.Parameter(torch.randn(self.nneighbor, self.nkeypoints, self.nkeypoints))
                else:
                    self.affinity_params = nn.Parameter(
                        torch.randn(self.nneighbor, self.nkeypoints, self.nkeypoints - 1))


            else:
                if self.affinity_ver < 3:
                    self.affinity_params = nn.Parameter(torch.zeros(self.nneighbor, self.nkeypoints, self.nkeypoints))
                else:
                    self.affinity_params = nn.Parameter(torch.ones(self.nneighbor, self.nkeypoints, self.nkeypoints - 1))


    def anneal(self, nepoch):
        if self.keypoints_graph == 'affinity_params':
            if self.affinity_anneal > nepoch:
                self.affinity_params.requires_grad = False
            else:
                if not self.affinity_start:
                    self.affinity_start = True
                    self.affinity_params.requires_grad = True


    def forward(self, seq, Tcond=None):
        B, T, C, *X = seq.size()
        heatmaps, keypoints, gaussians, first_feature = self.vox_to_kypt(seq, Tcond=Tcond)
        # heatmaps, keypoints, first_feature = self.vox_to_kypt(seq)
        # gaussians = refine_gaussian_map(gaussians, affinity)
        # gaussians = edge_gaussian_map(keypoints, affinity, self.sigmas, self.grid_size)
        recon = self.kypt_to_vox(gaussians, first_feature, seq[:, 0])
        # losses
        # recon loss
        sum_dim = [2, ] + [i for i in range(3, len(X) + 3)]
        criterion = nn.BCELoss(reduction='none')
        recon_loss = criterion(recon, seq).mean(dim=sum_dim)  # (B, T)

        # anneal_reg
        # kypt_const_loss = abs(keypoints[..., -1].mean(dim=1, keepdim=True) - keypoints[..., -1]).mean(dim=-1)
        kypt_const_loss = torch.zeros(B, T).to(seq.device)

        # sparsity_loss
        sparsity_loss = get_keypoint_sparsity_loss(heatmaps)  # (B, T) sum along K

        # separation_loss
        separation_loss = get_temporal_separation_loss(keypoints, self.sep_sigma)  # (B, T) sum along K

        # volume_fitting_loss
        if self.fixed_sigma:
            sigmas = self.vox_to_kypt.sigmas
        else:
            sigmas = torch.sigmoid(self.vox_to_kypt.sigmas) * self.vox_to_kypt.max_sigma
        vol_fit_reg = get_volume_fitting_loss(seq, keypoints, sigmas, self.vol_fit_type)

        # graph-related
        if self.keypoints_graph == 'none' or not self.affinity_start:
            affinity=None
            local_const_loss = torch.zeros(B, T).to(seq.device)
            time_const_loss, sparsity_const_loss, intensity_const_loss = torch.zeros_like(local_const_loss), torch.zeros_like(local_const_loss), torch.zeros_like(local_const_loss)
            graph_traj_loss = torch.zeros(B, T).to(seq.device)
            graph_vol_loss = torch.zeros(B, T).to(seq.device)
        elif self.keypoints_graph == 'affinity_params':
            affinity = self.get_affinity()
            if self.keypoints_detach:
                local_const_loss, time_const_loss, sparsity_const_loss, intensity_const_loss = get_graph_consistency_loss(keypoints.detach(), affinity,
                                                                    local_const=self.using_local_const,
                                                                    time_const=self.using_time_const,
                                                                    sparsity_const=self.using_sparsity_const,
                                                                    intensity_const=self.using_intensity_const,
                                                                    ver=self.graph_loss_ver)
                if self.using_graph_traj:
                    graph_traj_loss = get_graph_traj_loss(keypoints.detach(), affinity, ver=self.graph_loss_ver)
                else:
                    graph_traj_loss = torch.zeros(B, T).to(seq.device)


                graph_vol_loss = torch.zeros(B, T).to(seq.device)
            else:
                local_const_loss, time_const_loss, sparsity_const_loss, intensity_const_loss = get_graph_consistency_loss(keypoints, affinity,
                                                                    local_const=self.using_local_const,
                                                                    time_const=self.using_time_const,
                                                                    sparsity_const=self.using_sparsity_const,
                                                                    intensity_const=self.using_intensity_const,
                                                                    ver=self.graph_loss_ver
                                                                    )
                if self.using_graph_traj:
                    graph_traj_loss = get_graph_traj_loss(keypoints, affinity, ver=self.graph_loss_ver)
                else:
                    graph_traj_loss = torch.zeros(B, T).to(seq.device)


                graph_vol_loss = torch.zeros(B, T).to(seq.device)

        things = dict(
            recon=recon,
            keypoints=keypoints,
            heatmaps=heatmaps,
            affinity=affinity,
            recon_loss=recon_loss.mean(),
            vol_fit_reg=vol_fit_reg.mean(),
            kypt_const_loss=kypt_const_loss.mean(),
            separation_loss=separation_loss.mean(),
            sparsity_loss=sparsity_loss.mean(),
            local_const_loss=local_const_loss.mean(),
            time_const_loss=time_const_loss.mean(),
            sparsity_const_loss=sparsity_const_loss.mean(),
            intensity_const_loss=intensity_const_loss.mean(),
            graph_traj_loss=graph_traj_loss.mean(),
            graph_vol_loss=graph_vol_loss.mean(),
            first_feature=first_feature,  #(B, D, G, ..., G)
        )

        return things

    def get_affinity(self):
        # ver 0.0
        if self.affinity_ver == 0:
            W = torch.softmax(self.affinity_params, dim=2)  # (nneigbor, K, K)

        # ver 1.0
        elif self.affinity_ver == 1:
            softplus = nn.Softplus()
            W = softplus(self.affinity_params)
            W = torch.stack([torch.matmul(W[k], W[k].T) for k in range(self.nneighbor)], dim=0)  # (nneighbor, K, K)
            W = W * (1 - torch.eye(self.nkeypoints).to(W.device)[None])
            W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)  # normalize

        # ver 2.0
        elif self.affinity_ver == 2:
            softplus = nn.Softplus()
            W = softplus(self.affinity_params)
            W = W * (1 - torch.eye(self.nkeypoints).to(W.device)[None])
            W = torch.softmax(W, dim=2)  # (nneigbor, K, K)

        elif self.affinity_ver == 3:
            W_temp = torch.softmax(self.affinity_params, dim=-1)

            W = []
            for n in range(self.nneighbor):
                m_up = torch.cat([torch.zeros(self.nkeypoints, 1).to(W_temp.device), torch.triu(W_temp[n], diagonal=0)], dim=-1)
                m_low = torch.cat([torch.tril(W_temp[n], diagonal=-1), torch.zeros(self.nkeypoints, 1).to(W_temp.device)], dim=-1)
                W.append(m_up + m_low)
            W = torch.stack(W, dim=0)
        elif self.affinity_ver == 4:
            W_temp = F.gumbel_softmax(self.affinity_params, tau=1.0, dim=-1)
            W = []
            for n in range(self.nneighbor):
                m_up = torch.cat([torch.zeros(self.nkeypoints, 1).to(W_temp.device), torch.triu(W_temp[n], diagonal=0)], dim=-1)
                m_low = torch.cat([torch.tril(W_temp[n], diagonal=-1), torch.zeros(self.nkeypoints, 1).to(W_temp.device)], dim=-1)
                W.append(m_up + m_low)
            W = torch.stack(W, dim=0)
        else:
            raise ValueError("Invalid affinity version")

        return W.unsqueeze(-1)

    def decode_from_dyna(self, keypoints, first_feature, first_frame):
        '''
        :param keypoints: (B, Tgen, K, D + 1)
        :param first_feature: (B, feat_dim, G, .., G)
        :param first_frame: (B, 1, X1, X2, ..., XD)
        :return:
        '''
        _, Tgen, *_ = keypoints.size()
        gaussians = []
        for t in range(Tgen):
            keypoint = keypoints[:, t]
            gaussian = []
            for k in range(self.nkeypoints):
                keypoint_k = keypoint[:, k].unsqueeze(1)
                gaussian_k = extract_gaussian_map_from_keypoints(keypoint_k, sigma=self.sigmas[k], G=self.grid_size // 4)
                gaussian.append(gaussian_k)
            gaussian = torch.cat(gaussian, dim=1)
            gaussians.append(gaussian)
        gaussians = torch.stack(gaussians, dim=1)
        # affinity = self.get_affinity()
        # gaussians = refine_gaussian_map(gaussians, affinity)
        # gaussians = edge_gaussian_map(keypoints, affinity, self.sigmas, self.grid_size)
        gen = self.kypt_to_vox(gaussians, first_feature, first_frame)

        things = dict(
           gen=gen
        )

        return things


class VoxToKyptNet(nn.Module):
    """
    Extract K keypoints from given observation
    """
    def __init__(self, grid_size, nkeypoints, input_dim, sigmas, fixed_sigma, const_intensity):
        nn.Module.__init__(self)

        # config
        self.grid_size = grid_size
        self.feat_dim = 128
        self.nkeypoints = nkeypoints
        self.fixed_sigma = fixed_sigma
        self.const_intensity = const_intensity

        if self.fixed_sigma:
            self.sigmas = sigmas
        else:
            self.max_sigma = sigmas[0] * 2.0
            self.sigmas = nn.Parameter(torch.randn(self.nkeypoints))

        def _build_feature_net(in_channels, out_channels):
            return nn.Sequential(
                Basic3DBlock(1 + in_channels, out_channels // 4, 5),
                Pool3DBlock(2, out_channels // 4),
                Res3DBlock(out_channels // 4, out_channels // 2),
                Pool3DBlock(2, out_channels // 2),
                HG(out_channels // 2, out_channels // 2, N=self.grid_size // 4),
                Res3DBlock(out_channels // 2, out_channels)
            )
        def _build_heatmap_net(in_channels, out_channels, act='softplus'):
            if act == 'softplus':
                act_layer = nn.Softplus()
            elif act == 'leakyrelu':
                act_layer = nn.LeakyReLU()
            return nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                act_layer)

        self.extract_features = _build_feature_net(input_dim, self.feat_dim)
        if not self.const_intensity:
            self.extract_heatmaps_from_features = _build_heatmap_net(self.feat_dim, self.nkeypoints, act='softplus')
        else:
            self.extract_heatmaps_from_features = _build_heatmap_net(self.feat_dim, self.nkeypoints, act='leakyrelu')

        if self.const_intensity:
            heatmap_grid = [self.grid_size // 4] * 3
            if self.const_intensity == 1:
                self.initial_heatmaps = nn.Parameter(torch.randn(self.nkeypoints, *heatmap_grid))
            elif self.const_intensity == 2 or self.const_intensity == 3 or self.const_intensity == 4:
                self.extract_spatio_temporal_features = _build_feature_net(input_dim, self.feat_dim * 2)
                self.extract_spatio_temporal_heatmaps_from_features = _build_heatmap_net(self.feat_dim * 2, self.nkeypoints, act='leakyrelu')
            self.propagate_heatmaps = nn.Sequential(nn.Conv3d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0),
                                                    # nn.BatchNorm3d(1, track_running_stats=False),
                                                    nn.Softplus())

    def forward(self, seq, Tcond=None):

        B, T, C, *X = seq.size()

        if self.fixed_sigma:
            sigmas = self.sigmas
        else:
            sigmas = torch.sigmoid(self.sigmas) * self.max_sigma

        if self.const_intensity:
            if self.const_intensity == 1:
                prev_heatmap = self.initial_heatmaps[None].expand(B, -1, -1, -1, -1)
            elif self.const_intensity == 2 or self.const_intensity == 3:
                seq_summed = seq.mean(dim=1)
                obs = add_coord_channels(seq_summed)
                feature = self.extract_spatio_temporal_features(obs)
                heatmap = self.extract_spatio_temporal_heatmaps_from_features(feature)
                prev_heatmap = heatmap
            elif self.const_intensity == 4:
                if Tcond is not None:
                    seq_summed = 1 - seq.mean(dim=1) + 1 / T  # dynamic 1/T ~ static 1 --> dynamic 1 ~ static 1/T
                else:
                    seq_summed = 1 - seq[:, :Tcond].mean(dim=1) + 1 / Tcond  # in case for generation time in rl
                seq_summed = seq_summed * (seq.sum(dim=1).clip(0, 1))
                obs = add_coord_channels(seq_summed)
                feature = self.extract_spatio_temporal_features(obs)
                heatmap = self.extract_spatio_temporal_heatmaps_from_features(feature)
                prev_heatmap = heatmap
        heatmaps = []
        keypoints = []
        gaussians = []
        for t in range(T):
            obs = seq[:, t]
            obs = add_coord_channels(obs)
            feature = self.extract_features(obs)
            if t == 0:
                first_feature = feature
            heatmap = self.extract_heatmaps_from_features(feature)

            if self.const_intensity:
                _, _, *G = heatmap.size()
                heatmap = heatmap.view(B * self.nkeypoints, 1, *G)
                prev_heatmap = prev_heatmap.reshape(B * self.nkeypoints, 1, *G)
                heatmap = self.propagate_heatmaps(torch.cat([heatmap, prev_heatmap], dim=1))
                heatmap = heatmap.view(B, self.nkeypoints, *G)
                if self.const_intensity == 1 or self.const_intensity == 2:
                    prev_heatmap = heatmap

            keypoint = extract_keypoints_from_heatmap(heatmap)
            gaussian = []
            for k in range(self.nkeypoints):
                keypoint_k = keypoint[:, k].unsqueeze(1)
                gaussian_k = extract_gaussian_map_from_keypoints(keypoint_k, sigma=sigmas[k], G=self.grid_size // 4)
                gaussian.append(gaussian_k)
            gaussian = torch.cat(gaussian, dim=1)

            heatmaps.append(heatmap)
            keypoints.append(keypoint)
            gaussians.append(gaussian)

        heatmaps = torch.stack(heatmaps, dim=1)  # (B, T, nkeypoints, G, G, G)
        keypoints = torch.stack(keypoints, dim=1)  # (B, T, nkeypoints, D + 1)
        gaussians = torch.stack(gaussians, dim=1)


        return heatmaps, keypoints, gaussians, first_feature
        # return heatmaps, keypoints, first_feature



class KyptToVoxNet(nn.Module):
    def __init__(self, grid_size, nkeypoints, input_dim, gaussian_cat_type):
        nn.Module.__init__(self)
        self.grid_size = grid_size
        self.output_map_width = self.grid_size // 4
        assert self.grid_size % self.output_map_width == 0
        self.feat_dim = 128
        self.nkeypoints = nkeypoints
        self.gaussian_cat_type = gaussian_cat_type


        def _build_adjust_net(nkeypoints):
            return nn.Sequential(nn.Conv3d(in_channels=self.feat_dim + 2 * nkeypoints + input_dim, out_channels=self.feat_dim, kernel_size=1),
                                 # nn.BatchNorm3d(self.feat_dim, track_running_stats=False),
                                 nn.LeakyReLU())

        self.adjust_combined_representation = _build_adjust_net(self.nkeypoints)
        self.decode_voxel_from_combined_representation = self.build_voxel_decoder()

    def forward(self, gaussians, first_feature, first_frame, sharpness=10.0, translation=0.5):
        '''
            keypoints : (B, T, K, D + 1)
            first_feature : (B, feat_dim, self.output_map_width, self.output_map_width)
            first_frame : (B, C, X1, X2, ... )
        '''
        B, T, K, *_ = gaussians.size()
        if self.gaussian_cat_type == 'max':
            gaussians = gaussians.max(dim=2, keepdim=True).values
            gaussians = torch.cat([gaussians] * K, dim=2)
        elif self.gaussian_cat_type == 'sum':
            gaussians = gaussians.sum(dim=2, keepdim=True).clip(0, 1)
            gaussians = torch.cat([gaussians] * K, dim=2)

        vox_recons = []

        for t in range(T):

            combined_representation = torch.cat([gaussians[:, t], first_feature, gaussians[:, 0]], dim=1)  # (B, 2K, G, ..., )
            combined_representation = add_coord_channels(combined_representation)  # (B, feat_dim + 2K + D, G, ..., )
            combined_representation = self.adjust_combined_representation(combined_representation)  # (B, feat_dim, G, G, ..)
            vox_recon = self.decode_voxel_from_combined_representation(combined_representation)
            vox_recon = torch.sigmoid(sharpness * (torch.tanh(vox_recon) + first_frame - translation))  # act(1) = 0.99330 act(0) = 0.006692
            vox_recons.append(vox_recon)

        vox_recons = torch.stack(vox_recons, dim=1)

        return vox_recons

    def build_voxel_decoder(self):
        """Decodes images from feature maps.
        The decoder iteratively doubles the resolution, and halves the number of
        filters until the size of the feature maps the same with original image.
        shape of keypoint sequence: [T, 16, K, 3]
        """

        layers = []
        # interpolate (16, 16) -> (32, 32), size perserving & feature dim reducing convs
        layers.append(
            nn.Upsample(scale_factor =2.0, mode='trilinear',
                        align_corners=False))
        layers.append(nn.Conv3d(in_channels=self.feat_dim, out_channels=self.feat_dim // 2, kernel_size=3, stride=1,
                                padding=1))
        # layers.append(nn.BatchNorm3d(self.feat_dim // 2, track_running_stats=False))
        layers.append(nn.GroupNorm(self.feat_dim // (2 * 16), self.feat_dim // 2))
        layers.append(nn.LeakyReLU())
        layers.append(
            nn.Conv3d(in_channels=self.feat_dim // 2, out_channels=self.feat_dim // 2, kernel_size=3, stride=1,
                      padding=1))  # (32, 32, 64)
        # layers.append(nn.BatchNorm3d(self.feat_dim // 2, track_running_stats=False))
        layers.append(nn.GroupNorm(self.feat_dim // (2 * 16), self.feat_dim // 2))
        layers.append(nn.LeakyReLU())
        layers.append(
            nn.Upsample(scale_factor =2.0, mode='trilinear',
                        align_corners=False))
        layers.append(
            nn.Conv3d(in_channels=self.feat_dim // 2, out_channels=self.feat_dim // 4, kernel_size=3, stride=1,
                      padding=1))
        # layers.append(nn.BatchNorm3d(self.feat_dim // 4, track_running_stats=False))
        layers.append(nn.GroupNorm(self.feat_dim // (4 * 16), self.feat_dim // 4))
        layers.append(nn.LeakyReLU())
        layers.append(
            nn.Conv3d(in_channels=self.feat_dim // 4, out_channels=self.feat_dim // 4, kernel_size=3, stride=1,
                      padding=1))  # (64, 64, 32)
        # layers.append(nn.BatchNorm3d(self.feat_dim // 4, track_running_stats=False))
        layers.append(nn.GroupNorm(self.feat_dim // (4 * 16), self.feat_dim // 4))
        layers.append(nn.LeakyReLU())

        # adjust to original channels
        layers.append(nn.Conv3d(in_channels=self.feat_dim // 4, out_channels=1, kernel_size=1))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)















