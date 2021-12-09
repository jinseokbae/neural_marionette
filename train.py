from __future__ import print_function
import argparse
import sys
import torch
import torch.utils.data
import shutil
import os
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(1,'.')

from model.neural_marionette import NeuralMarionette
from utils.train_utils import COLORS, display_opts, display_it, weights_init, LOGGER, LOSS_SCHEDULER
from dataset.dataset import DATASET_LIST
from vis.visualize import vis_recon, vis_keypoints
from utils.eval_utils import evaluate, evaluate_final

from dataset.config import adjust_config
import collections
torch.autograd.set_detect_anomaly(True)
#========================================================================================
#                                   argument parsing
#========================================================================================
parser = argparse.ArgumentParser()
# about training itself
parser.add_argument('--seed', type=int, default= 0, help='seed for random')
parser.add_argument('--nepoch', type=int, default = 2000, help='')
parser.add_argument('--lrate', type=float, default =1e-3, help='')
parser.add_argument('--firstdecay', type=int, default = 1,  help='')
parser.add_argument('--seconddecay', type=int, default = 10,  help='')
parser.add_argument('--resume_epoch', type=str, default='0',help='')
parser.add_argument('--max_grad_norm', type=float, default = 30.0,  help='')
parser.add_argument('--device', type=str, default='cuda:0')

# about saving & logging
parser.add_argument('--training_id', type=str, default=None,help='')
parser.add_argument('--save_every', type=int, default=1, help='epochs')
parser.add_argument('--save_que_len', type=int, default=100, help='maximum number of pth files')
parser.add_argument('--log_every', type=int, default=1, help='epochs')
parser.add_argument('--exp_name', type=str, default='default', help='')
parser.add_argument('--log_gif_num', type=int, default=8, help='sequence to log with tensorboard')
parser.add_argument('--log_gif_every', type=int, default=1, help='')

# about dataset
parser.add_argument('--dataset', type=str, default ="dfaust", help='')
parser.add_argument('--nbatch', type=int, default=24, help='')
parser.add_argument('--input_dim', type=int, default=3, help='2 for pixel, 3 for voxel')
parser.add_argument('--grid_size', type=int, default=64, help='grid size')
parser.add_argument('--is_binarized', type=int, default=1, help='whether dataset is binarized or sdf')
parser.add_argument('--Ttot', type=int, default=10, help='conditioned frame number')
parser.add_argument('--Tcond', type=int, default=5, help='conditioned frame number')
parser.add_argument('--sample_rate', type=int, default=1, help='conditioned frame number')
parser.add_argument('--random_crop', type=int, default=1, help='conditioned frame number')
parser.add_argument('--surface_sampled', type=int, default=1, help='conditioned frame number')
parser.add_argument('--debug', type=int, default=0, help='conditioned frame number')
parser.add_argument('--is_eval', type=int, default=0, help='conditioned frame number')

# about architecture
parser.add_argument('--nkeypoints', type=int, default=22, help='number of keypoints')
parser.add_argument('--gaussian_sigma', type=float, default=1.5, help='sigma for gaussian')
parser.add_argument('--dyna_module', type=str, default='SVRNN', help='choice of dynamics module')
parser.add_argument('--nlatent_kypt', type=int, default=128, help='kypt latent dimension')
parser.add_argument('--nhidden_kypt', type=int, default=512, help='kypt rnn state dimension')
parser.add_argument('--sep_sigma', type=float, default=0.02, help='kypt separation sigma dimension')

# loss weights
parser.add_argument('--recon_weight', type=float, default=100.0, help='')
parser.add_argument('--sparse_weight', type=float, default=5.0, help='')
parser.add_argument('--sep_weight', type=float, default=0.1, help='')
parser.add_argument('--vol_reg_weight', type=float, default=10.0, help='')
parser.add_argument('--kypt_const_weight', type=float, default=0.0, help='')
parser.add_argument('--local_const_weight', type=float, default=1e-3, help='')
parser.add_argument('--time_const_weight', type=float, default=1.0, help='')
parser.add_argument('--sparsity_const_weight', type=float, default=0.01, help='')
parser.add_argument('--intensity_const_weight', type=float, default=0.01, help='')
parser.add_argument('--graph_traj_weight', type=float, default=1.0, help='')
parser.add_argument('--graph_vol_weight', type=float, default=0.0, help='')
parser.add_argument('--kypt_recon_weight', type=float, default=1.0, help='')
parser.add_argument('--kl_kypt_weight', type=float, default=0.003, help='')
parser.add_argument('--gae_recon_weight', type=float, default=1.0, help='')
parser.add_argument('--topo_recon_weight', type=float, default=0.01, help='')

# anneal-related
parser.add_argument('--detector_start', type=int, default=0, help='')
parser.add_argument('--affinity_anneal', type=int, default=0, help='')
parser.add_argument('--learner_start', type=int, default=1e9, help='')
parser.add_argument('--detector_end', type=int, default=-1, help='')
parser.add_argument('--learner_end', type=int, default=-1, help='')

# pretraining-related
parser.add_argument('--pretrained_mode', type=int, default=0, help='0 for not using any, 1 for using pretrained detector only, 2 for using pretrained detector, dyna')
parser.add_argument('--pretrained_dir', type=str, default='pretrained', help='')

# experimental - detector
parser.add_argument('--vol_fit_type', type=str, default='chamfer', help='')
parser.add_argument('--gaussian_cat_type', type=str, default='none', help='')
parser.add_argument('--fixed_sigma', type=int, default=1, help='')
parser.add_argument('--keypoints_graph', type=str, default='affinity_params', help='')
parser.add_argument('--nneighbor', type=int, default=2, help='')
parser.add_argument('--keypoints_detach', type=int, default=0, help='')
parser.add_argument('--graph_random_init', type=int, default=0, help='')
parser.add_argument('--using_local_const', type=int, default=1, help='')
parser.add_argument('--using_time_const', type=int, default=1, help='')
parser.add_argument('--using_sparsity_const', type=int, default=1, help='')
parser.add_argument('--using_intensity_const', type=int, default=1, help='')
parser.add_argument('--const_intensity', type=int, default=3, help='')
parser.add_argument('--affinity_ver', type=int, default=3, help='')
parser.add_argument('--graph_loss_ver', type=int, default=1, help='')

# experimental - dynamics learner
parser.add_argument('--transition_type', type=str, default='dl', help='')
parser.add_argument('--using_pose_feature', type=int, default=1, help='')
parser.add_argument('--nlatent_pose', type=int, default=32, help='')
parser.add_argument('--using_dim_enhance', type=int, default=1, help='')
parser.add_argument('--enhance_dim', type=int, default=16, help='')
parser.add_argument('--sharing_enc_net', type=int, default=-1, help='version -1 : no enc, version 0 : no sharing, version 1 : sharing, version 2 : sharing but detach')
parser.add_argument('--state_mode', type=str, default='no_cat', help='')
parser.add_argument('--action_mode', type=str, default='pose', help='pose or vel')
parser.add_argument('--appnp_alpha', type=float, default=0.3, help='')

# experimental - agent related
parser.add_argument('--ncontrols', type=int, default=5, help='')
parser.add_argument('--replay_size', type=int, default=4e3, help='')
parser.add_argument('--agent_gamma', type=float, default=0.99, help='decaying factor for reward.')
parser.add_argument('--agent_alpha', type=float, default=0.2, help='temperature for policy entropy')
parser.add_argument('--agent_polyak', type=float, default=0.995, help='polyak param for target netowork update')
parser.add_argument('--rod_init_mode', type=str, default='static_uniform', help='')
parser.add_argument('--mapping_mode', type=str, default='node', help='')
parser.add_argument('--start_step', type=int, default=500, help='')

opt = parser.parse_args()
opt = adjust_config(opt)

# make network deterministic
torch.backends.cudnn.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)

if opt.pretrained_mode == 0:
    opt.training_id = "rl_setup/disc_training/%s/%s/%dkypt" %(opt.dataset, opt.keypoints_graph, opt.nkeypoints)
elif opt.pretrained_mode == 1:
    opt.training_id = "rl_setup/dyna_training/%s/%s/%s/%dkypt/%dzkypt_%dhkypt" %(opt.dataset, opt.transition_type, opt.dyna_module,
                                      opt.nkeypoints, opt.nlatent_kypt, opt.nhidden_kypt)
    opt.detector_end = 0
    opt.learner_start = 0
elif opt.pretrained_mode == 2:
    opt.training_id = "rl_setup/rl_training/%s/%s/%s/%dkypt_%dctrl" %(opt.dataset, opt.dyna_module, opt.mapping_mode,
                                                                      opt.nkeypoints, opt.nctrols)
    opt.detector_end = 0
    opt.learner_end = 0
else:
    raise ValueError("Wrong assignment on argument pretrained_mode!")

if opt.log_gif_num > opt.nbatch:
    opt.log_gif_num = opt.nbatch

display_opts(opt)
#========================================================================================


#========================================================================================
#                                   training element
#========================================================================================
DATASET = DATASET_LIST()                                       #list of all the methods
#========================================================================================

#========================================================================================
#                                       List of Loss
#========================================================================================

LOSS_LIST = ['recon_loss', 'sparsity_loss', 'separation_loss', 'vol_fit_reg', 'kypt_const_loss',
             'local_const_loss', 'time_const_loss', 'sparsity_const_loss', 'intensity_const_loss', 'graph_traj_loss', 'graph_vol_loss',
             'kl_kypt', 'kypt_recon_loss', 'gae_recon_loss', 'topo_recon_loss']

LOSS_WEIGHTS = {'recon_loss' : opt.recon_weight, 'sparsity_loss': opt.sparse_weight, 'separation_loss': opt.sep_weight,
                'vol_fit_reg': opt.vol_reg_weight, 'kypt_const_loss': opt.kypt_const_weight,
                'local_const_loss': opt.local_const_weight, 'time_const_loss' : opt.time_const_weight, 'sparsity_const_loss': opt.sparsity_const_weight, 'intensity_const_loss': opt.intensity_const_weight,
                'graph_traj_loss': opt.graph_traj_weight, 'graph_vol_loss': opt.graph_vol_weight,
                'kypt_recon_loss': opt.kypt_recon_weight, 'kl_kypt': opt.kl_kypt_weight, 'gae_recon_loss': opt.gae_recon_weight, 'topo_recon_loss': opt.topo_recon_weight}


detector_time = (opt.detector_start, opt.detector_end)
learner_time = (opt.learner_start, opt.learner_end)

ANNEAL_EPOCHS = {'recon_loss':detector_time, 'sparsity_loss':detector_time, 'separation_loss':detector_time,
                 'vol_fit_reg': detector_time, 'kypt_const_loss': detector_time,
                 'local_const_loss': detector_time, 'time_const_loss' : detector_time, 'sparsity_const_loss': detector_time, 'intensity_const_loss': detector_time,
                 'graph_traj_loss':detector_time, 'graph_vol_loss': detector_time,
                 'kl_kypt':learner_time, 'kypt_recon_loss':learner_time, 'gae_recon_loss': learner_time, 'topo_recon_loss' : learner_time}

MODULE_ACTIVE_EPOCHS = {'detector': detector_time, 'learner': learner_time}

for loss_name in LOSS_LIST:
    if loss_name in LOSS_WEIGHTS.keys():
        continue
    LOSS_WEIGHTS[loss_name] = 1.0  # fill with one for other loss


loss_scheduler = LOSS_SCHEDULER(LOSS_LIST, LOSS_WEIGHTS, ANNEAL_EPOCHS, MODULE_ACTIVE_EPOCHS)
#========================================================================================


#========================================================================================
#                                    dataset selection
#========================================================================================
if opt.dataset not in DATASET.type:
    print(COLORS.FAIL,"ERROR please select the dataset from : ",COLORS.ENDC)
    for dataset in DATASET.type:
        print("   >",dataset)
    exit()

dataset_train = DATASET.load(training=True, options=opt)
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               shuffle=True,
                                               batch_size=opt.nbatch,
                                               num_workers=0)

dataset_valid = DATASET.load(training=False, options=opt)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                               shuffle=False,
                                               batch_size=opt.nbatch,
                                               num_workers=0)

#======================================================================================



#========================================================================================
#                                    model selection
#========================================================================================
network = NeuralMarionette(opt).cuda()                                      # load the loss
logger_path = "./output/%s/%s" % (opt.training_id, opt.exp_name)
ckpt_path = os.path.join(logger_path, 'epochs')
start_epoch = 0

if os.path.exists(ckpt_path):
    resumes = os.listdir(ckpt_path)
    resume_file = os.path.join(ckpt_path, opt.resume_epoch, 'network.pth')
    buffer_file = os.path.join(ckpt_path, opt.resume_epoch, 'replay_buffer.pkl')
    # demanding resume_epoch exists
    if os.path.exists(resume_file) and opt.resume_epoch != '0':
        checkpoint = torch.load(resume_file)
        network.load_state_dict(checkpoint)
        if opt.pretrained_model == 2:
            with open(buffer_file, 'rb') as f:
                buffer = pickle.load(f)
            network.agent.replay_buffer.load(buffer)
        start_epoch = int(opt.resume_epoch) + 1

# by default, resuming from most latest checkpoint
    elif int(opt.resume_epoch) == 0 and len(resumes) > 0:
        resumes = sorted([int(ep) for ep in resumes])
        resume_epoch = resumes[-1]
        resume_file = os.path.join(ckpt_path, str(resume_epoch), 'network.pth')
        checkpoint = torch.load(resume_file)
        network.load_state_dict(checkpoint)
        start_epoch = resume_epoch + 1
    else:
        if opt.pretrained_mode == 0:
            network.apply(weights_init)  # init the weight
# in case where previous experiments do not exist or resume_epoch number does not exist
elif int(opt.resume_epoch) != 0:
    raise ValueError('No previous checkpoints from this setting.')

elif opt.pretrained_mode == 0:
    network.apply(weights_init)  # init the weight

if opt.pretrained_mode == 1:
    resume_file = os.path.join(opt.pretrained_dir, 'detector', '%s_detector.pth' % opt.dataset)
    if os.path.exists(resume_file):
        checkpoint = torch.load(resume_file)
        checkpoint = collections.OrderedDict(filter(lambda p: p[0].split('.')[0] == 'kypt_detector', checkpoint.items()))
        checkpoint = collections.OrderedDict({k[14:]: v for k, v in checkpoint.items()})  # substract 'kypt_detector.'
        network.kypt_detector.load_state_dict(checkpoint)
    else:
        raise ValueError('pretrained file is not existing.')
elif opt.pretrained_mode == 2:
    resume_file = os.path.join(opt.pretrained_dir, 'dyna',  '%s_%s_%s_%s_dyna.pth' % (opt.dataset, opt.dyna_module, opt.transition_type, opt.action_mode))
    if os.path.exists(resume_file):
        checkpoint = torch.load(resume_file)
        # load kypt_detector part
        checkpoint_detector = collections.OrderedDict(filter(lambda p: p[0].split('.')[0] == 'kypt_detector', checkpoint.items()))
        checkpoint_detector = collections.OrderedDict({k[14:]: v for k, v in checkpoint_detector.items()})  # substract 'kypt_detector.'
        network.kypt_detector.load_state_dict(checkpoint_detector)
        # load dyna_module part
        checkpoint_dyna = collections.OrderedDict(filter(lambda p: p[0].split('.')[0] == 'dyna_module', checkpoint.items()))
        checkpoint_dyna = collections.OrderedDict({k[12:]: v for k, v in checkpoint_dyna.items()})  # substract 'dyna_module.'
        network.dyna_module.load_state_dict(checkpoint_dyna)
        network.agent.trainset_len = len(dataset_train)
    else:
        raise ValueError('pretrained file is not existing.')

#========================================================================================




#======================================================================================
#                                 optimizer
#======================================================================================
optimizer = torch.optim.Adam(network.parameters(), lr=opt.lrate)
#======================================================================================


#======================================================================================
#                               training logs
#======================================================================================

if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists(logger_path):
    os.makedirs(logger_path)
    os.makedirs(os.path.join(logger_path, 'epochs'))

with open('%s/opt.pickle'%logger_path, 'wb') as handle:
    pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

train_log = LOGGER()
valid_log = LOGGER()

for loss_name in LOSS_LIST:
    train_log.add_keys(loss_name)
    if loss_name != 'nrom_cons':
        valid_log.add_keys(loss_name)

if opt.is_eval:
    scores = dict()
    scores_log = dict()
    eval_metrics = ['semantic']
    for eval_name in eval_metrics:
        valid_log.add_keys(eval_name)
        scores[eval_name] = None
        scores_log[eval_name] = None


#======================================================================================

#======================================================================================
#                                   tensorboard
#======================================================================================
os.makedirs(os.path.join(logger_path, 'logs'), exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(logger_path, 'logs'), purge_step=start_epoch, flush_secs=30)
os.makedirs(os.path.join(logger_path, 'gifs'), exist_ok=True)
#======================================================================================
#                                   TRAINING BEGIN
#======================================================================================
for epoch_id in range(start_epoch, opt.nepoch):

    network.train()
    dataset_train.log_epoch(epoch_id)
    dataset_valid.log_epoch(epoch_id)
    # scheduling loss
    loss_scheduler.anneal(epoch_id)
    if epoch_id % opt.log_gif_every == 0:
        loss_scheduler.display_active()
        loss_scheduler.display_loss()
    #==================================================================================
    #                                   training
    #==================================================================================
    stacked_voxel = None
    network.anneal(epoch_id)
    network.control_active(loss_scheduler.module_actives)
    if epoch_id < opt.firstdecay:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=opt.lrate)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, network.parameters()), opt.max_grad_norm)
    elif epoch_id >= opt.firstdecay and epoch_id < opt.seconddecay:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()),lr = opt.lrate/4)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, network.parameters()), opt.max_grad_norm/4)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()),lr = opt.lrate/10)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, network.parameters()), opt.max_grad_norm/10)

    for batch_id, batch in enumerate(dataloader_train):
        network.anneal(epoch_id, batch_id)
        if type(batch) == list:
            batch = [b.cuda() for b in batch]
            voxel, _ = batch
            # xs.requires_grad_(True)
            # xo.requires_grad_(True)
        else:
            batch = batch.cuda()
            voxel = batch

        optimizer.zero_grad()
        log = network(voxel, module_actives=loss_scheduler.module_actives)
        loss = 0
        for loss_name in LOSS_LIST:
            if loss_name in loss_scheduler.current_loss_names:
                loss += LOSS_WEIGHTS[loss_name] * log[loss_name]
            elif loss_name in log.keys():
                loss += 0 * log[loss_name]
            else:
                log[loss_name] = torch.tensor(0.0).to(voxel.device)
                loss += log[loss_name]
            train_log.add(loss_name, log[loss_name].item())
        loss.backward()
        '''
        #### when you want to monitor gradient norm
        total_norm = 0
        for p in filter(lambda p: p.requires_grad, network.parameters()):

            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        '''
        optimizer.step()


        display_it("train", "total loss", opt, epoch_id, batch_id, loss.item())

    #==================================================================================


    #==================================================================================
    #                                validation
    #==================================================================================
    if opt.pretrained_mode == 0:
        log_voxel, log_pred_voxel, log_keypoints, log_pred_gen, log_keypoints_gen = None, None, None, None, None
        log_affinity = None
        log_y_prims, log_gt_frames = None, None
    elif opt.pretrained_mode == 1 or opt.pretrained_mode == 2:
        log_voxel, log_pred_voxel, log_keypoints, log_keypoints_recon, log_keypoints_gen = None, None, None, None, None
        log_affinity, log_A, log_A_hats, log_A_hats_gen, log_pred_gen, = None, None, None, None, None

    first_flag = False
    with torch.no_grad():

        network.eval()

        for batch_id, batch in enumerate(dataloader_valid):

            if type(batch) == list:
                batch = [b.cuda() for b in batch]
                voxel, gt_kypt = batch
            else:
                batch = batch.cuda()
                voxel = batch

            log = network(voxel, module_actives=loss_scheduler.module_actives)
            if not first_flag and opt.pretrained_mode == 0: # logging first batch
                log_voxel = voxel
                log_pred_voxel = log['recon'] if 'recon' in log.keys() else None
                log_keypoints = log['keypoints'] if 'keypoints' in log.keys() else None
                log_y_prims = log['y_prims'] if 'y_prims' in log.keys() else None
                log_affinity = log['affinity'] if 'affinity' in log.keys() else None
                log_gt_frames = log['gt_frames'] if 'gt_frames' in log.keys() else None
                log_g = network.generate(voxel, module_actives=loss_scheduler.module_actives)
                log_pred_gen = log_g['gen'] if 'gen' in log_g.keys() else None
                log_keypoints_gen = log_g['keypoints'] if 'keypoints' in log_g.keys() else None
                first_flag = True
            elif not first_flag and (opt.pretrained_mode == 1 or opt.pretrained_mode == 2):
                log_voxel = voxel
                log_keypoints = log['keypoints'] if 'keypoints' in log.keys() else None
                log_keypoints_recon = log['kypt_recon'] if 'kypt_recon' in log.keys() else None
                log_affinity = log['affinity'] if 'affinity' in log.keys() else None
                log_A = log['A'] if 'A' in log.keys() else None
                log_A_hats = log['A_hats'] if 'A_hats' in log.keys() else None
                log_g = network.generate(voxel, module_actives=loss_scheduler.module_actives)
                log_pred_gen = log_g['gen'] if 'gen' in log_g.keys() else None
                log_keypoints_gen = log_g['keypoints'] if 'keypoints' in log_g.keys() else None
                log_A_hats_gen = log_g['A_hats'] if 'A_hats' in log_g.keys() else None
                first_flag = True
            loss = 0

            for loss_name in LOSS_LIST:
                if loss_name == 'norm_cons':
                    continue
                if loss_name in loss_scheduler.current_loss_names:
                    loss += LOSS_WEIGHTS[loss_name] * log[loss_name]
                elif loss_name in log.keys():
                    loss += 0 * log[loss_name]
                else:
                    log[loss_name] = torch.tensor(0.0).to(voxel.device)
                    loss += log[loss_name]
                valid_log.add(loss_name, log[loss_name].item())


            display_it("valid", "total loss", opt, epoch_id, batch_id, loss.item())

            # ==================================================================================
            if opt.is_eval:
                params = dict(
                    keypoints=log['keypoints'].clone(),
                    gt_keypoints=gt_kypt.detach()
                )
                for eval_name in eval_metrics:
                    temp_log = evaluate(eval_name, scores, params)
                    scores[eval_name] = temp_log['scores']
                    scores_log[eval_name] = temp_log['scores_log']
                    display_it("eval", eval_name, opt, epoch_id, batch_id, scores_log[eval_name])
                    valid_log.add(eval_name, scores_log[eval_name].item())

    #==================================================================================

    #==================================================================================
    #                          saving loss and parameters
    #==================================================================================
    train_log.reset()
    valid_log.reset()



    '''
    tensorboard logging
    '''
    # scalars
    if epoch_id % opt.log_every == 0:
        for loss_name in LOSS_LIST:
            writer.add_scalar('train/%s' % loss_name, train_log.history[-1][loss_name], epoch_id)
            writer.add_scalar('valid/%s' % loss_name, valid_log.history[-1][loss_name], epoch_id)
        if opt.is_eval:
            for eval_name in eval_metrics:
                writer.add_scalar('eval/%s' % eval_name, valid_log.history[-1][eval_name], epoch_id)

    # visualizing
    if opt.pretrained_mode == 0:
        if epoch_id % opt.log_gif_every == 0 or epoch_id < 10:
            with torch.no_grad():
                gif_recon, gif_keypoints, gif_graphs = None, None, None
                if log_pred_voxel is not None and log_keypoints is not None:
                    gif_recon = vis_recon(
                        vox=log_voxel,
                        recon=log_pred_voxel,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        log_num=opt.log_gif_num,
                        group='track'
                    )
                    gif_keypoints = vis_keypoints(
                        vox=log_voxel,
                        keypoints=log_keypoints,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        affinity=log_affinity,
                        log_num=opt.log_gif_num,
                        group='track'
                    )


                for i in range(opt.log_gif_num):
                    if gif_recon is not None:
                        writer.add_video(f'track/recon_{i}', gif_recon[i][None], epoch_id)
                    if gif_keypoints is not None:
                        writer.add_video(f'track/keypoints_{i}', gif_keypoints[i][None], epoch_id)

                gif_recon, gif_keypoints, gif_graphs = None, None, None
                if log_pred_gen is not None and log_keypoints_gen is not None:
                    gif_recon = vis_recon(
                        vox=log_voxel,
                        recon=log_pred_gen,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        log_num=opt.log_gif_num,
                        group='gen',
                        Tcond=opt.Tcond
                    )
                    gif_keypoints = vis_keypoints(
                        vox=log_voxel,
                        keypoints=log_keypoints_gen,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        affinity=log_affinity,
                        log_num=opt.log_gif_num,
                        group='gen',
                        Tcond=opt.Tcond
                    )

                for i in range(opt.log_gif_num):
                    if gif_recon is not None:
                        writer.add_video(f'gen/recon_{i}', gif_recon[i][None], epoch_id)
                    if gif_keypoints is not None:
                        writer.add_video(f'gen/keypoints_{i}', gif_keypoints[i][None], epoch_id)

    elif opt.pretrained_mode == 1 or opt.pretrained_mode == 2:
        if epoch_id % opt.log_gif_every == 0:
            with torch.no_grad():
                gif_keypoints_recon, gif_graph_recon = None, None
                gif_keypoints = vis_keypoints(
                    vox=log_voxel,
                    keypoints=log_keypoints,
                    logger_path=logger_path,
                    nepoch=epoch_id,
                    affinity=log_affinity,
                    log_num=opt.log_gif_num,
                    group='track'
                )
                if log_keypoints_recon is not None:
                    gif_keypoints_recon = vis_keypoints(
                        vox=log_voxel,
                        keypoints=log_keypoints_recon,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        affinity=log_affinity,
                        log_num=opt.log_gif_num,
                        group='track'
                    )
                    gif_keypoints_recon = torch.cat([gif_keypoints, gif_keypoints_recon], dim=-1)

                if log_A is not None and log_A_hats is not None:

                    gif_keypoints_A = vis_keypoints(
                        vox=log_voxel,
                        keypoints=log_keypoints_recon,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        affinity=log_A,
                        log_num=opt.log_gif_num,
                        group='track',
                        mode='A'
                    )
                    gif_keypoints_A_hats = vis_keypoints(
                        vox=log_voxel,
                        keypoints=log_keypoints_recon,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        affinity=log_A_hats,
                        log_num=opt.log_gif_num,
                        group='track',
                        mode='A_hats'
                    )
                    gif_graph_recon = torch.cat([gif_keypoints_A, gif_keypoints_A_hats], dim=-1)

                for i in range(opt.log_gif_num):
                    if gif_keypoints_recon is not None:
                        writer.add_video(f'track/kypt_recon_{i}', gif_keypoints_recon[i][None], epoch_id)
                    if gif_graph_recon is not None:
                        writer.add_video(f'track/graph_recon_{i}', gif_graph_recon[i][None], epoch_id)

                gif_keypoints_recon, gif_recon = None, None
                if log_keypoints_gen is not None:
                    gif_keypoints_recon = vis_keypoints(
                        vox=log_voxel,
                        keypoints=log_keypoints_gen,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        affinity=log_affinity,
                        log_num=opt.log_gif_num,
                        group='gen',
                        Tcond=opt.Tcond
                    )
                    gif_keypoints_recon = torch.cat([gif_keypoints, gif_keypoints_recon], dim=-1)

                if log_pred_gen is not None:
                    gif_recon = vis_recon(
                        vox=log_voxel,
                        recon=log_pred_gen,
                        logger_path=logger_path,
                        nepoch=epoch_id,
                        log_num=opt.log_gif_num,
                        group='gen',
                        Tcond=opt.Tcond
                    )
                for i in range(opt.log_gif_num):
                    if gif_recon is not None:
                        writer.add_video(f'gen/recon_{i}', gif_recon[i][None], epoch_id)
                    if gif_keypoints_recon is not None:
                        writer.add_video(f'gen/kypt_recon_{i}', gif_keypoints_recon[i][None], epoch_id)
    '''
    saving checkpoints
    '''
    if epoch_id % opt.save_every == 0:
        # save path files
        if len(os.listdir(os.path.join(logger_path, 'epochs'))) == opt.save_que_len:
            ckpt_list = os.listdir(os.path.join(logger_path, 'epochs'))
            ckpt_list.sort(key=lambda x: int(x))
            shutil.rmtree(os.path.join(logger_path, 'epochs', ckpt_list[0]))
            if os.path.exists(os.path.join(logger_path, 'gifs', ckpt_list[0])):
                shutil.rmtree(os.path.join(logger_path, 'gifs', ckpt_list[0]))
        os.makedirs(os.path.join(logger_path, 'epochs', str(epoch_id)), exist_ok=True)
        torch.save(network.state_dict(), '%s/epochs/%d/network.pth' % (logger_path, epoch_id))
    #==================================================================================i
