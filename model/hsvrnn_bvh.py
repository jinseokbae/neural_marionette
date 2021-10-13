import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from utils.dyna_utils import process_affinity_glob
from utils.geo_utils import compute_global_rot_from_local_rot


class HSVRNNBVH(nn.Module):

    def __init__(self, options):
        nn.Module.__init__(self)
        self.nkeypoints = options.nkeypoints
        self.nlatent_kypt = options.nlatent_kypt
        self.nhidden_kypt = options.nhidden_kypt
        self.input_dim = options.input_dim  # (2 for 2d 3 for 3d)
        self.transition_type = options.transition_type
        self.state_mode = options.state_mode
        self.action_mode = options.action_mode

        state_dim = self.nkeypoints * (self.input_dim + 1)

        # action dimension
        if self.transition_type == 'dl':
            act_dim = 0

        # post / prior network
        self.extract_post_dist = nn.Sequential(
            nn.Linear(in_features=self.nhidden_kypt + state_dim + act_dim,
                      out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=self.nlatent_kypt * 2)  # mu, sigma
        )
        self.extract_prior_dist = nn.Sequential(
            nn.Linear(in_features=self.nhidden_kypt + act_dim,
                      out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=self.nlatent_kypt * 2)  # mu, sigma
        )
        self.root_intensity_decoder = nn.Sequential(  # root position
            nn.Linear(in_features=self.nhidden_kypt + self.nlatent_kypt,
                      out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=3 + self.nkeypoints),
            nn.Tanh()
        )

        self.joint_matrix_decoder = nn.Sequential(  # rotation params(6) and intensity(1)
            nn.Linear(in_features=self.nhidden_kypt + self.nlatent_kypt,
                      out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=6 * self.nkeypoints),
        )

        # rnn-related
        self.kypt_rnn_cell = nn.GRUCell(input_size=state_dim + self.nlatent_kypt,
                                        hidden_size=self.nhidden_kypt)

        self.init_kypt_rnn_state = nn.Parameter(torch.randn(1, self.nhidden_kypt))

        self.A, self.priority, self.parents = None, None, None

        self.offset_param = nn.Parameter(torch.randn(self.nkeypoints, 3))
        self.offset_param.requires_grad = False

    def encode(self, keypoints, affinity, SAMPLE_NUM=10):
        """
            keypoints : (B, T, K, D + 1) -> detached
        """
        B, T, K, _ = keypoints.size()

        prev_state = self.init_kypt_rnn_state.expand(B, -1)

        if self.A is None:
            A, priority, parents = process_affinity_glob(affinity)  # each matrix has dimension of (B, T, K, K)
            self.A = A.float()
            self.priority = priority
            self.parents = parents

        keypoints_inferred = []
        kl_kypt = []
        z_kypts = []
        h_kypts = [prev_state]
        R_inferred = []
        # offset
        offset = self.get_offset(keypoints)

        for t in range(T):
            keypoint = keypoints[:, t]
            # prior net
            if self.transition_type == 'dl':
                params_prior = self.extract_prior_dist(torch.cat([prev_state], dim=-1))  # (B, Dh) -> (B, Zk + Zk)
            prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 1e-4
            z_kypt_prior_dist = Normal(prior_mean, prior_std)

            # posterior net
            keypoint_flat = keypoint.view(B, -1)  # (B, K * (D + 1))
            if self.transition_type == 'dl':
                params_post = self.extract_post_dist(torch.cat([prev_state, keypoint_flat], dim=-1))
            post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
            post_std = F.softplus(post_std) + 1e-4
            z_kypt_post_dist = Normal(post_mean, post_std)

            # sample to find closest latent from detected keypoints (of detector) (only for training)
            z_kypt_sampled = z_kypt_post_dist.rsample(sample_shape=(SAMPLE_NUM,))  # (SAMPLE_NUM, B, Zk)
            keypoint_sampled_flat_list = []
            R_list = []
            for i in range(SAMPLE_NUM):
                keypoint_sampled_flat, R = self.extract_kypt_from_latent_and_state(
                    torch.cat([prev_state, z_kypt_sampled[i]], dim=-1), offset)
                keypoint_sampled_flat_list.append(keypoint_sampled_flat)
                R_list.append(R)
            keypoint_sampled_flats = torch.stack(keypoint_sampled_flat_list, dim=0)  # (SAMPLE_NUM, B, K * (D + 1))
            keypoint_distance = (keypoint_flat[None] - keypoint_sampled_flats).pow(2).sum(-1)  # (SAMPLE_NUM, B)
            min_sample_idx = keypoint_distance.argmin(dim=0)  # (B, )
            batch_idx = torch.arange(0, B, dtype=torch.int64).to(keypoint.device)
            best_latent = z_kypt_sampled[min_sample_idx, batch_idx]
            best_keypoint_flat = keypoint_sampled_flats[min_sample_idx, batch_idx]

            R = torch.stack(R_list, dim=0)  # (SAMPLE_NUM, B, K, 3, 3)
            best_R = R[min_sample_idx, batch_idx]

            # update
            # don't give detected keypoints to rnn
            rnn_input = torch.cat([best_keypoint_flat, best_latent], dim=-1)
            prev_state = self.kypt_rnn_cell(rnn_input, prev_state)

            # list fill
            kl_kypt.append(kl_divergence(z_kypt_post_dist, z_kypt_prior_dist))  # (B, )
            keypoints_inferred.append(best_keypoint_flat.view(B, K, -1))  # (B, K, D + 1)
            R_inferred.append(best_R)
            z_kypts.append(best_latent)
            h_kypts.append(prev_state)

        kl_kypt = torch.stack(kl_kypt, dim=1)  # (B, T, D)
        keypoints_inferred = torch.stack(keypoints_inferred, dim=1)  # (B, T, K, D + 1)
        R_inferred = torch.stack(R_inferred, dim=1)
        z_kypts = torch.stack(z_kypts, dim=1)  # (B, T, Dz)
        h_kypts = torch.stack(h_kypts, dim=1)  # (B, T + 1, Dh)

        kypt_recon_loss = (keypoints_inferred - keypoints).pow(2).sum(dim=(2, 3))  # (B, T)

        things = dict(
            kypt_recon=keypoints_inferred[..., :4],
            R=R_inferred,
            z_kypts=z_kypts,
            h_kypts=h_kypts,
            kl_kypt=kl_kypt.mean(),
            kypt_recon_loss=kypt_recon_loss.mean(),
            gae_recon_loss=torch.tensor(0).to(keypoints.device),
            topo_recon_loss=torch.tensor(0).to(keypoints.device)
        )

        return things

    def generate(self, keypoints_cond, affinity=None, Ttot=10, Tcond=3, SAMPLE_NUM=10):
        '''
            keypoints_cond : (B, Tcond, K, D + 1) when self.transition_type == 'dl'

            Tgen : T - Tcond
        '''

        B, _, K, _ = keypoints_cond.size()

        prev_state = self.init_kypt_rnn_state.expand(B, -1)

        keypoints_inferred = []
        offset = self.get_offset(keypoints_cond)
        for t in range(Tcond):
            keypoint = keypoints_cond[:, t]

            # posterior net
            keypoint_flat = keypoint.view(B, -1)  # (B, K * (D + 1))
            if self.transition_type == 'dl':
                params_post = self.extract_post_dist(torch.cat([prev_state, keypoint_flat], dim=-1))
            post_mean, post_std = torch.chunk(params_post, 2, dim=-1)
            post_std = F.softplus(post_std) + 1e-4
            z_kypt_post_dist = Normal(post_mean, post_std)

            # sample to find closest latent from detected keypoints (of detector) (only for training)
            z_kypt_sampled = z_kypt_post_dist.rsample(sample_shape=(SAMPLE_NUM,))  # (SAMPLE_NUM, B, Zk)
            keypoint_sampled_flat_list = []
            for i in range(SAMPLE_NUM):
                keypoint_sampled_flat, _ = self.extract_kypt_from_latent_and_state(
                    torch.cat([prev_state, z_kypt_sampled[i]], dim=-1), offset)
                keypoint_sampled_flat_list.append(keypoint_sampled_flat)
            keypoint_sampled_flats = torch.stack(keypoint_sampled_flat_list, dim=0)  # (SAMPLE_NUM, B, K * (D + 1))
            keypoint_distance = (keypoint_flat[None] - keypoint_sampled_flats).pow(2).sum(-1)  # (SAMPLE_NUM, B)
            min_sample_idx = keypoint_distance.argmin(dim=0)  # (B, )
            batch_idx = torch.arange(0, B, dtype=torch.int64).to(keypoint.device)
            best_latent = z_kypt_sampled[min_sample_idx, batch_idx]
            best_keypoint_flat = keypoint_sampled_flats[min_sample_idx, batch_idx]

            # update
            # don't give detected keypoints to rnn
            rnn_input = torch.cat([best_keypoint_flat, best_latent], dim=-1)
            prev_state = self.kypt_rnn_cell(rnn_input, prev_state)

            # list fill
            keypoints_inferred.append(best_keypoint_flat.view(B, K, -1))  # (B, K, D + 1)

        keypoints_inferred = torch.stack(keypoints_inferred, dim=1)  # (B, Tcond, K, D + 1)
        # prevs_state is preserved to generation time

        keypoints_generated = []
        for t in range(Tcond, Ttot):
            # prior net
            if self.transition_type == 'dl':
                params_prior = self.extract_prior_dist(torch.cat([prev_state], dim=-1))  # (B, Dh) -> (B, Zk + Zk)
            prior_mean, prior_std = torch.chunk(params_prior, 2, dim=-1)
            prior_std = F.softplus(prior_std) + 1e-4
            z_kypt_prior_dist = Normal(prior_mean, prior_std)

            z_kypt_sampled = z_kypt_prior_dist.rsample()
            keypoint_sampled_flat, _ = self.extract_kypt_from_latent_and_state(
                torch.cat([prev_state, z_kypt_sampled], dim=-1), offset)

            # update
            rnn_input = torch.cat([keypoint_sampled_flat, z_kypt_sampled], dim=-1)
            prev_state = self.kypt_rnn_cell(rnn_input, prev_state)

            # list fill
            keypoints_generated.append(keypoint_sampled_flat.view(B, K, -1))

        keypoints_generated = torch.stack(keypoints_generated, dim=1)  # (B, Tgen, K, D + 1)

        things = dict(
            keypoints_cond=keypoints_inferred[..., :4],
            keypoints_gen=keypoints_generated[..., :4],
        )

        return things

    def get_offset(self, keypoints):
        B, T, K, _ = keypoints.size()
        pos = keypoints[..., :3].clone()
        dist = (pos[:, :, :, None] - pos[:, :, None]).pow(2).sum(dim=-1).sqrt()

        med_dist = dist.median(dim=1).values  # (B, K, K)
        offset_scale = []
        for k in range(K):
            parent = self.parents[k]
            offset_scale.append(med_dist[:, k, parent])
        offset_scale = torch.stack(offset_scale, dim=-1)  # (B, K)

        offset_param = self.offset_param
        offset_norm = offset_param / (offset_param.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-10)

        offset = offset_norm[None].expand(B, -1, -1).contiguous() * offset_scale[..., None]

        return offset[..., None].detach()

    def extract_kypt_from_latent_and_state(self, decoder_input, offset):
        '''
        :param decoder_input: (B, D+H)
        :param offset: (B, K, 3, 1)
        :return:
        '''
        # parents (K, )
        B, _ = decoder_input.size()

        raw = self.root_intensity_decoder(decoder_input)  # (B, 3)
        root_pos = raw[:, :3].contiguous()
        intensity = ((raw[:, 3:] + 1) * 0.5).unsqueeze(-1).contiguous()  # (B, K, 1)s

        rot_params = self.joint_matrix_decoder(decoder_input)
        rot_params = rot_params.reshape(B, self.nkeypoints, -1).contiguous()
        R = compute_global_rot_from_local_rot(rot_params, self.priority, self.parents)  # (B, K, 3, 3)

        pos = torch.zeros(B, self.nkeypoints, 3).to(decoder_input.device)
        root = self.priority.indices[0]
        pos[:, root] = root_pos

        for idx in self.priority.indices[1:]:
            pos[:, idx] = torch.bmm(R[idx.item()], offset[:, idx]).squeeze(-1) + pos[:, self.parents[idx]]

        processed = torch.cat([pos, intensity], dim=-1)

        Rs = []
        for idx in range(self.nkeypoints):
            Rs.append(R[idx])
        Rs = torch.stack(Rs, dim=1)  # (B, K, 3, 3)

        return processed.view(B, -1), Rs








