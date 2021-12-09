from __future__ import print_function
import numpy as np
import pickle
import torch
import sys
import os
import random
from torch import nn

class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LOSS_SCHEDULER:
    def __init__(self, LOSS_LIST, LOSS_WEIGHTS, ANNEAL_EPOCHS, MODULE_ACTIVE_EPOCHS):
        '''
        :param LOSS_LIST: list of loss names
        :param LOSS_WEIGHTS: list of loss weights
        :param ANNEAL_EPOCHS: dictionary of loss_name(str, key)-(start, end)(tuple, val)
                                if end==-1, it means use until end
        '''
        self.loss_names = LOSS_LIST
        self.loss_weights = LOSS_WEIGHTS
        checkpoints = set()
        checkpoints.add(0)
        checkpoints.add(float("inf"))
        for loss_name in ANNEAL_EPOCHS.keys():
            start, end = ANNEAL_EPOCHS[loss_name]
            if start >= 0:
                checkpoints.add(start)
            if end >= 0:
                checkpoints.add(end)
        self.milestones = sorted(list(checkpoints))  # [0, m1, m2, ..., mN, end(inf)]
        self.loss_names_anneal = [[] for _ in range(len(self.milestones) - 1)] #[(0 ~ m1), (m1 ~ m2), ..., (mN ~ end)]


        for idx, m_start in enumerate(self.milestones[:-1]):
            m_end = self.milestones[idx + 1]
            for loss_name in self.loss_names:
                start, end = ANNEAL_EPOCHS[loss_name]
                if (start <= m_start) and (end >= m_end or end == -1):
                    self.loss_names_anneal[idx].append(loss_name)

        self.module_actives = {key : False for key in MODULE_ACTIVE_EPOCHS.keys()}
        self.module_active_epochs = MODULE_ACTIVE_EPOCHS
        for module_name in MODULE_ACTIVE_EPOCHS.keys():
            interval = self.module_active_epochs[module_name]
            if interval[0] == -1:
                interval = (float("inf"), float("inf"))
            if interval[1] == -1:
                interval = (interval[0], float("inf"))
            self.module_active_epochs[module_name] = interval
        self.display_anneal()
        self.current_loss_names = None

    def display_anneal(self):
        display_msg = "ANNEALS:\n"
        for idx, m_start in enumerate(self.milestones[:-1]):
            m_end = self.milestones[idx + 1]
            if m_end != float("inf"):
                display_msg += "\t%03d~%03d\n" % (m_start, m_end)
            else:
                display_msg += "\t%03d~end\n" % (m_start)
            for loss_name in self.loss_names_anneal[idx]:
                display_msg += "\t%s %s %s\n" % (COLORS.WARNING, loss_name, COLORS.ENDC,)
        print(display_msg)

    def display_loss(self):
        display_msg = "LOSSES OPTIMIZED:\n"
        for loss_name in self.current_loss_names:
            display_msg += "\t%s %s %s\n" % (COLORS.WARNING, loss_name, COLORS.ENDC)
        print(display_msg)

    def display_active(self):
        display_msg = "\n\nMODULE ACTIVES:\n"
        for module_name in self.module_actives.keys():
            color = COLORS.OKBLUE if self.module_actives[module_name] else COLORS.FAIL
            display_msg += "\t%10s %s %s %s\n" % (module_name, color, bool(self.module_actives[module_name]), COLORS.ENDC)
        print(display_msg)

    def anneal(self, epoch_id):
        for module_name in self.module_active_epochs.keys():
            start_epoch, end_epoch = self.module_active_epochs[module_name]
            if start_epoch <= epoch_id and epoch_id < end_epoch and not self.module_actives[module_name]:
                self.module_actives[module_name] = True
            elif epoch_id >= end_epoch and self.module_actives[module_name]:
                self.module_actives[module_name] = False
        for idx, m_start in enumerate(self.milestones[:-1]):
            m_end = self.milestones[idx + 1]
            if epoch_id >= m_start and epoch_id < m_end:
                self.current_loss_names = self.loss_names_anneal[idx]




def display_opts(opts):


    display_msg = """PARAMETERS:
            training_id   %s %s %s
            exp_name      %s %s %s
            resume_epoch  %s %s %s
            nbatch        %s %s %s
            grid_size     %s %s %s
            Ttot          %s %s %s
            Tcond         %s %s %s
            nkeypoints    %s %s %s
            dyna_module   %s %s %s
            recon_w       %s %s %s
            sparse_w      %s %s %s
            sep_w         %s %s %s
            vol_reg_w     %s %s %s
            local_const_w %s %s %s
            time_const_w  %s %s %s
            spars_const_w %s %s %s
            inten_const_w %s %s %s
            graph_traj_w  %s %s %s
            kypt_recon_w  %s %s %s
            kl_kypt_w     %s %s %s
            gae_recon_w   %s %s %s
            topo_recon_w  %s %s %s
            """%(COLORS.OKGREEN,opts.training_id,COLORS.ENDC,
                 COLORS.OKGREEN,opts.exp_name,COLORS.ENDC,
                 COLORS.OKGREEN, opts.resume_epoch, COLORS.ENDC,

                 COLORS.OKBLUE, opts.nbatch, COLORS.ENDC,
                 COLORS.OKBLUE, opts.grid_size , COLORS.ENDC,
                 COLORS.OKBLUE, opts.Ttot, COLORS.ENDC,
                 COLORS.OKBLUE, opts.Tcond, COLORS.ENDC,
                 COLORS.OKBLUE, opts.nkeypoints, COLORS.ENDC,
                 COLORS.OKBLUE, opts.dyna_module, COLORS.ENDC,

                 COLORS.WARNING, opts.recon_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.sparse_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.sep_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.vol_reg_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.local_const_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.time_const_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.sparsity_const_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.intensity_const_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.graph_traj_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.kypt_recon_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.kl_kypt_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.gae_recon_weight, COLORS.ENDC,
                 COLORS.WARNING, opts.topo_recon_weight, COLORS.ENDC,
                 )

    print(display_msg)


def display_it(mode, name, opt, epoch_id, batch_id, loss, print_every=200):
    """display iteration"""

    if batch_id % print_every == 0:
        msg = ''

        if mode == 'train':
            msg = "[%s%s - %s%s] - %d/%d - %04d   %s%f%s" % (COLORS.OKGREEN,
                                                        opt.exp_name,
                                                        name,
                                                        COLORS.ENDC,
                                                        epoch_id,
                                                        opt.nepoch,
                                                        batch_id,
                                                        COLORS.BOLD,
                                                        loss,
                                                        COLORS.ENDC)

        if mode == 'valid':
            msg = "[%s%s - %s%s] - %d/%d - %04d   %s%f%s" % (COLORS.OKBLUE,
                                                        opt.exp_name,
                                                        name,
                                                        COLORS.ENDC,
                                                        epoch_id,
                                                        opt.nepoch,
                                                        batch_id,
                                                        COLORS.BOLD,
                                                        loss,
                                                        COLORS.ENDC)

        if mode == 'eval':
            msg = "[%s%s - %s%s] - %d/%d - %04d   %s%f%s" % (COLORS.WARNING,
                                                        opt.exp_name,
                                                        name,
                                                        COLORS.ENDC,
                                                        epoch_id,
                                                        opt.nepoch,
                                                        batch_id,
                                                        COLORS.BOLD,
                                                        loss,
                                                        COLORS.ENDC)
        print(msg)

def display_agent(var_counts):
    msg = """
    %sNumber of agent parameters:%s
        %spi : %d%s
        %sq1 : %d%s
        %sq2 : %d%s 
    """%(
        COLORS.OKGREEN, COLORS.ENDC,
        COLORS.OKBLUE, var_counts[0], COLORS.ENDC,
        COLORS.OKBLUE, var_counts[1], COLORS.ENDC,
        COLORS.OKBLUE, var_counts[2], COLORS.ENDC,
    )
    print(msg)


class LOGGER:
    """logger of the network loss """

    def __init__(self):
        self.history = []
        self.data = {}
        self.keys = []
    def add_keys(self, key):
        self.data[key] = []
        self.keys.append(key)

    def add(self, key, val):
        self.data[key].append(val)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def mean(self, key):
        m = np.mean(np.array(self.data[key]))
        return m

    def reset(self):
        new_hist = {}
        for key in self.keys:
            if self.data[key]:
                new_hist[key] = np.mean(np.array(self.data[key]))
        self.history.append(new_hist)
        self.data = {}
        for key in self.keys:
            self.data[key] = []


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'GraphConv':
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Block') != -1:
        for mm in m.modules():
            if isinstance(mm, nn.Conv3d):
                mm.weight.data.normal_(0.0, 0.001)
                mm.bias.data.fill_(0)
            elif isinstance(mm, nn.ConvTranspose3d):
                mm.weight.data.normal_(0.0, 0.001)
                mm.bias.data.fill_(0)