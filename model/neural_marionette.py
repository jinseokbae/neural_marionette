import torch
import torch.nn as nn
from model.kypt_detector import KyptDetector
from model.hsvrnn_bvh import HSVRNNBVH

class NeuralMarionette(nn.Module):
    def __init__(self, options=None):
        super(NeuralMarionette, self).__init__()
        self.options = options
        self.kypt_detector = KyptDetector(options)
        self.Tcond = self.options.Tcond
        self.dyna_module = HSVRNNBVH(options)

        self.current_actives = {'detector': True, 'learner': True}
        self.transition_type = options.transition_type


    def anneal(self, nepoch, nbatch=None):
        if nbatch is None: # anneal only once per epoch
            self.kypt_detector.anneal(nepoch)

    def control_active(self, module_actives):
        for module_name in self.current_actives.keys():
            if self.current_actives[module_name] != module_actives[module_name]:
                if module_name == 'detector':
                    module = self.kypt_detector
                elif module_name == 'learner':
                    module = self.dyna_module
                for child in module.modules():
                    for param in child.parameters():
                        param.requires_grad = module_actives[module_name]
                self.current_actives[module_name] = module_actives[module_name]

    def forward(self, vox_seq, module_actives=None):
        '''
        seq : # (B, T, N, C), cuda tensor
        module_actives : dictionary of module_active (True or False)
        '''
        log = dict()
        if module_actives['detector']:
            kypt_detector_log = self.kypt_detector(vox_seq)
            keypoints = kypt_detector_log['keypoints']
            affinity = kypt_detector_log['affinity'] if 'affinity' in kypt_detector_log.keys() else None
            log.update(kypt_detector_log)
        elif not module_actives['detector'] and module_actives['learner']:
            with torch.no_grad():
                kypt_detector_log = self.kypt_detector(vox_seq)
                keypoints = kypt_detector_log['keypoints']
                affinity = kypt_detector_log['affinity'] if 'affinity' in kypt_detector_log.keys() else None
                log.update(kypt_detector_log)

        if module_actives['learner']:
            dyna_module_log = self.dyna_module.encode(keypoints.detach(), affinity.detach())
            log.update(dyna_module_log)

        return log

    def generate(self, vox_seq, module_actives=None):
        '''
        :param seq: input seq
        :return:
        '''
        B, T, C, *X = vox_seq.size()
        assert self.Tcond < T

        log = dict()
        if module_actives['learner']:
            # for conditioned step
            if self.transition_type == 'dl':
                kypt_detector_log = self.kypt_detector(vox_seq[:, :self.Tcond].contiguous())
            elif self.transition_type == 'rl':
                kypt_detector_log = self.kypt_detector(vox_seq)
            keypoints = kypt_detector_log['keypoints']  # (B, Tcond, K, D + 1)
            affinity = kypt_detector_log['affinity'] if 'affinity' in kypt_detector_log.keys() else None

            dyna_module_log = self.dyna_module.generate(keypoints, affinity, Ttot=T, Tcond=self.Tcond)

            first_feature = kypt_detector_log['first_feature']  # (B, feat_dim, G, .., G)
            keypoints_gen = dyna_module_log['keypoints_gen']  # (B, T - Tcond, K, D + 1)

            generated_log = self.kypt_detector.decode_from_dyna(keypoints_gen, first_feature, vox_seq[:, 0])

            # CONCAT
            # voxel
            recon = kypt_detector_log['recon'][:, :self.Tcond]  # (B, Tcond, 1, X1, ..., XD)
            gen = generated_log['gen']  # (B, T - Tcond, 1, X1, ..., XD)
            total_recon = torch.cat([recon, gen], dim=1)  # (B, T, 1, X1, ..., XD)

            # keypoints
            total_keypoints = torch.cat([keypoints[:, :self.Tcond], keypoints_gen], dim=1)  # (B, T, K, D + 1)
            if 'A_hats_cond' in dyna_module_log.keys():
                total_A_hats = torch.cat([dyna_module_log['A_hats_cond'], dyna_module_log['A_hats_gen']], dim=1)
            else:
                total_A_hats = None

            gen_log = dict(
                gen=total_recon,
                keypoints=total_keypoints,
                A_hats=total_A_hats
            )
            log.update(gen_log)

        return log
