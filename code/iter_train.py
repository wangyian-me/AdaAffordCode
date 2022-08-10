"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from sapien.core import Pose, ArticulationJointType
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from env import Env, ContactError, SVDError
from camera import Camera
from robots.panda_robot import Robot
import random
import utils
from data import SAPIENVisionDataset
from models import model_AIP, model_AAP, model_3d_w2a
import multiprocessing as mp
import time
parser = ArgumentParser()

parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--primact_type', type=str, default='pulling', help='the primact type')
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--AAP_dir', type=str, default='nothing')
parser.add_argument('--AAP_epoch', type=str, default='nothing')
parser.add_argument('--AIP_dir', type=str, default='nothing')
parser.add_argument('--AIP_epoch', type=str, default='nothing')
parser.add_argument('--out_dir', type=str, default='../logs')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--lr_decay_by', type=float, default=0.9)
parser.add_argument('--lr_decay_every', type=float, default=5000)
parser.add_argument('--step', type=int, default=4)
parser.add_argument('--sample_type', type=str, default='fps')
parser.add_argument('--data_dir', type=str)
args = parser.parse_args()

data_features = ['gripper_direction_world', 'gripper_action_dis', 'gt_motion', 'pcs', 'ctpt', 'joint_info',
                     'gripper_forward_direction_world', 'gt_labels', 'start_pos', 'end_pos', "shape_id", "mat44", "ctpt_2d", "cur_qpos", "trial_id","mu1","mu2","mu3","mu4","density","friction","rand_pos_seed"]
def get_hidden_info(input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list, joint_info_list, start_pos_list, end_pos_list, Encoder, conf, grad=False):
    dir = dir_list
    input_pcs = input_pcs_list
    ctpt = ctpt_list
    joint_info = joint_info_list
    f_dir = f_dir_list
    dir = torch.FloatTensor(np.array(dir)).to(conf.device)
    batch_size = dir.shape[0]
    dir = dir.view(batch_size, -1)
    ctpt = torch.FloatTensor(np.array(ctpt)).view(batch_size, -1).to(conf.device)
    joint_info = torch.FloatTensor(np.array(joint_info)).view(batch_size, -1).to(conf.device)
    f_dir = torch.FloatTensor(np.array(f_dir)).view(batch_size, -1).to(conf.device)
    # gt_labels = torch.FloatTensor(np.array(gt_labels)).view(batch_size).to(conf.device)
    input_pcs = torch.cat(input_pcs, dim=0).to(conf.device)  # B x 3N x 3   # point cloud
    # input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)  # B x 3N x 2
    batch_size = input_pcs.shape[0]

    # dis = dis_list
    dis = [float(i) for i in dis_list]
    # print("start_pos_list:", start_pos_list)
    start_pos = [float(i) for i in start_pos_list]
    end_pos = [float(i) for i in end_pos_list]
    push_dis = push_dis_list
    dis = torch.FloatTensor(np.array(dis)).view(batch_size, -1).to(conf.device)
    push_dis = torch.FloatTensor(np.array(push_dis)).view(batch_size, -1).to(conf.device)
    start_pos = torch.FloatTensor(np.array(start_pos)).view(batch_size, -1).to(conf.device)
    end_pos = torch.FloatTensor(np.array(end_pos)).view(batch_size, -1).to(conf.device)
    if grad == True:
        hidden_info = Encoder(dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir)
    else :
        with torch.no_grad():
            hidden_info = Encoder(dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir)

    return hidden_info

def critic_forward(batch, data_features, network, conf, hidden_info):
    
    dir = batch[data_features.index('gripper_direction_world')]
    input_pcs = batch[data_features.index('pcs')]
    ctpt = batch[data_features.index('ctpt')]
    joint_info = batch[data_features.index('joint_info')]
    f_dir = batch[data_features.index('gripper_forward_direction_world')]
    gt_labels = batch[data_features.index('gt_labels')]

    dir = torch.FloatTensor(np.array(dir)).to(conf.device)
    batch_size = dir.shape[0]
    dir = dir.view(batch_size, -1)
    hidden_info = hidden_info.repeat(batch_size, 1)
    # hidden_info = torch.FloatTensor(np.array(hidden_info)).view(batch_size, -1).to(conf.device)
    ctpt = torch.FloatTensor(np.array(ctpt)).view(batch_size, -1).to(conf.device)
    joint_info = torch.FloatTensor(np.array(joint_info)).view(batch_size, -1).to(conf.device)
    f_dir = torch.FloatTensor(np.array(f_dir)).view(batch_size, -1).to(conf.device)
    gt_labels = torch.FloatTensor(np.array(gt_labels)).view(batch_size).to(conf.device)
    input_pcs = torch.cat(input_pcs, dim=0).to(conf.device)  # B x 3N x 3   # point cloud
    # input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)  # B x 3N x 2
    batch_size = input_pcs.shape[0]
    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(
        -1)  # BN
    if conf.sample_type == 'fps':
        input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    elif conf.sample_type == 'random':
        pcs_id = ()
        for batch_idx in range(input_pcs.shape[0]):
            idx = np.arange(input_pcs[batch_idx].shape[0])
            np.random.shuffle(idx)
            while len(idx) < conf.num_point_per_shape:
                idx = np.concatenate([idx, idx])
            idx = idx[:conf.num_point_per_shape]
            pcs_id = pcs_id + (torch.tensor(np.array(idx)),)
        input_pcid2 = torch.stack(pcs_id, dim=0).long().reshape(-1)
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
    ctpt = ctpt.view(batch_size, -1).to(conf.device)
    input_pcs[:, 0] = ctpt

    start_pos = batch[data_features.index('start_pos')]
    start_pos = torch.FloatTensor(np.array(start_pos)).view(batch_size, -1).to(conf.device)
    # hidden_info = torch.FloatTensor(np.array(hidden_info)).view(batch_size, -1).to(conf.device)
    loss = network.critic.get_ce_loss(dir, input_pcs, ctpt, joint_info, f_dir, gt_labels, hidden_info, start_pos)

    return loss
def AIP_forward(batch, data_features, network, AAP, conf, input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list, joint_info_list, start_pos_list, end_pos_list):
    dir = dir_list
    input_pcs = input_pcs_list
    ctpt = ctpt_list
    joint_info = joint_info_list
    f_dir = f_dir_list
    # gt_labels = batch[data_features.index('gt_labels')]
    # hidden_info = batch[data_features.index('hidden_info')]
    dir = torch.FloatTensor(np.array(dir)).to(conf.device)
    batch_size = dir.shape[0]
    dir = dir.view(batch_size, -1)
    device = conf.device
    # hidden_info = torch.FloatTensor(np.array(hidden_info)).view(batch_size, -1).to(conf.device)
    ctpt = torch.FloatTensor(np.array(ctpt)).view(batch_size, -1).to(conf.device)
    joint_info = torch.FloatTensor(np.array(joint_info)).view(batch_size, -1).to(conf.device)
    f_dir = torch.FloatTensor(np.array(f_dir)).view(batch_size, -1).to(conf.device)
    # gt_labels = torch.FloatTensor(np.array(gt_labels)).view(batch_size).to(conf.device)
    input_pcs = torch.cat(input_pcs, dim=0).to(conf.device)  # B x 3N x 3   # point cloud
    # input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)  # B x 3N x 2
    batch_size = input_pcs.shape[0]

    # dis = dis_list
    dis = [float(i) for i in dis_list]
    # print("start_pos_list:", start_pos_list)
    start_pos = [float(i) for i in start_pos_list]
    end_pos = [float(i) for i in end_pos_list]
    push_dis = push_dis_list
    # start_pos = start_pos_list
    # end_pos = end_pos_list
    # print("dis:", dis)
    dis = torch.FloatTensor(np.array(dis)).view(batch_size, -1).to(conf.device)
    push_dis = torch.FloatTensor(np.array(push_dis)).view(batch_size, -1).to(conf.device)
    start_pos = torch.FloatTensor(np.array(start_pos)).view(batch_size, -1).to(conf.device)
    end_pos = torch.FloatTensor(np.array(end_pos)).view(batch_size, -1).to(conf.device)

    tot_hidden_info = []
    tot_score = []
    tot_c_score = []
    for i in range(1, batch_size):

        with torch.no_grad():
            prev_hidden_info = AAP.hidden_encoder(dir[:i], dis[:i], push_dis[:i], ctpt[:i], joint_info[:i],
                                                  input_pcs[:i], start_pos[:i], end_pos[:i], f_dir[:i]).detach().cpu().numpy()
            now_hidden_info = AAP.hidden_encoder(dir[:i+1], dis[:i+1], push_dis[:i+1], ctpt[:i+1], joint_info[:i+1],
                                                  input_pcs[:i+1], start_pos[:i+1], end_pos[:i+1], f_dir[:i+1]).detach().cpu().numpy()
        tot_hidden_info.append(prev_hidden_info)
        # ipdb.set_trace()
        PREV = torch.FloatTensor(np.array(prev_hidden_info)).to(conf.device)
        NOW = torch.FloatTensor(np.array(now_hidden_info)).to(conf.device)
        with torch.no_grad():
            score = critic_forward(batch, data_features, AAP, conf, PREV) - critic_forward(batch, data_features, AAP, conf, NOW)
            c_score = torch.sigmoid(AAP.critic(dir[i:i+1], input_pcs[i:i+1], ctpt[i:i+1], joint_info[i:i+1], f_dir[i:i+1], PREV,
                                             start_pos[i:i+1]))
        tot_score.append(score.item())
        # tot_a_score.append(a_score[0])
        tot_c_score.append(c_score[0])
        # print("score ", score.item(), "label", gt_labels[i].item(), "num", i)
    # ipdb.set_trace()
    tot_hidden_info = np.array(tot_hidden_info)
    tot_hidden_info = torch.from_numpy(tot_hidden_info).to(device)
    # tot_hidden_info = torch.cat(tot_hidden_info, dim=0).to(device)
    # tot_score = np.array(tot_score)
    # tot_score = torch.from_numpy(tot_score).to(device)
    # tot_score = torch.cat(tot_score, dim=0).to(device)
    tot_score = torch.tensor(tot_score).to(device)
    # tot_a_score = torch.tensor(tot_a_score).to(device).reshape(-1,1)
    tot_c_score = torch.tensor(tot_c_score).to(device).reshape(-1,1)
    # print(tot_c_score.shape)
    loss = network.get_loss(input_pcs[1:], tot_score, tot_hidden_info, dir[1:], f_dir[1:], tot_c_score)

    return loss
# load dataset

################################# load data ###################################
train_dataset = SAPIENVisionDataset([args.primact_type], data_features)
conf = args
conf.ignore_joint_info = True
conf.continuous = False

train_dataset.load_data(conf.data_dir, batch_size=conf.batch_size, ignore_joint_info=conf.ignore_joint_info)


train_dataset.get_seq()


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                   pin_memory=True, \
                                                   num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                                   worker_init_fn=utils.worker_init_fn)

def generate_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, movable_link_mask, mat44, device=None):
    num_point_per_shape = 10000
    out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
    mask = (out[:, :, 3] > 0.5)
    pc = out[mask, :3]
    movable_link_mask = movable_link_mask[mask]
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    while len(idx) < 30000:
        idx = np.concatenate([idx, idx])
    idx = idx[:30000]
    pc = pc[idx, :]
    movable_link_mask = movable_link_mask[idx]
    movable_link_mask = movable_link_mask.reshape(1, 30000, 1)
    pc[:, 0] -= 5

    world_pc = (mat44[:3, :3] @ pc.T).T
    pc = np.array(pc, dtype=np.float32)
    world_pc = np.array(world_pc, dtype=np.float32)
    pc = torch.from_numpy(pc).float().unsqueeze(0).to(device)
    world_pc = torch.from_numpy(world_pc).float().unsqueeze(0).to(device).contiguous()
    input_pcid1 = torch.arange(1).unsqueeze(1).repeat(1, num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(world_pc, num_point_per_shape).long().reshape(-1)  # BN

    # # random sample pts
    # pcs_id = ()
    # for batch_idx in range(pc.shape[0]):
    #     idx = np.arange(pc[batch_idx].shape[0])
    #     np.random.shuffle(idx)
    #     while len(idx) < 10000:
    #         idx = np.concatenate([idx, idx])
    #     idx = idx[:10000]
    #     pcs_id = pcs_id + (torch.tensor(np.array(idx)), )
    # input_pcid2 = torch.stack(pcs_id, dim=0).long().reshape(-1)

    pc = pc[input_pcid1, input_pcid2, :].reshape(1, num_point_per_shape, -1)  # 1 * N * 3
    world_pc = world_pc[input_pcid1, input_pcid2, :].reshape(1, num_point_per_shape, -1)
    movables = movable_link_mask[input_pcid1, input_pcid2.cpu().detach()]
    movables = movables.reshape(1, num_point_per_shape, 1)
    return world_pc, pc, movables


def bgs(d6s): # 3*2的矩阵(方向)，
    # print(d6s[0, 0, 0] *d6s[0, 0, 0] + d6s[0, 1, 0] * d6s[0, 1, 0] + d6s[0, 2, 0] *d6s[0, 2, 0])
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort

def get_aff_pic(init_pc_world, init_pc_cam, pc_cam, init_mask, tot_step, hidden_info, actor, AAP, mat44, start_pos, out_dir, batch_ind):
    actions = actor.inference_actor_whole_pc(init_pc_cam)
    actions = actions.detach().cpu().numpy()
    up = actions[:, :3]
    forward = actions[:, 3:]
    for i in range(10000):
        up[i] = mat44[:3, :3] @ up[i]
        forward[i] = mat44[:3, :3] @ forward[i]
    up = torch.from_numpy(up).to(args.device)
    forward = torch.from_numpy(forward).to(args.device)
    start_pos = torch.FloatTensor(np.array([start_pos])).to(args.device)
    state_joint_origins = torch.from_numpy(np.ones(40000)).float().view(-1, 4).to(args.device)
    with torch.no_grad():
        result = AAP.critic.inference_critic_score_diff_action(up, init_pc_world, None, state_joint_origins, forward, hidden_info, start_pos)
        result = result.detach().cpu().numpy()

    init_mask = init_mask.reshape(10000)
    result = result.reshape(10000)
    result *= init_mask
    fn = os.path.join(out_dir,f'{batch_ind}_{tot_step}_critic')
    utils.render_pts_label_png(fn, pc_cam, result)

out_dir = os.path.join(args.out_dir, args.exp_name)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(os.path.join(out_dir, 'ckpts')):
    os.mkdir(os.path.join(out_dir, 'ckpts'))

def run_an_collect(idx_process, args, transition_Q, epoch_Q):

    np.random.seed(random.randint(1, 1000) + idx_process)
    random.seed(random.randint(1, 1000) + idx_process + 7)

    device = args.device

    # out_dir = os.path.join(args.out_dir, args.exp_name)
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)

    # setup env
    flog = open(os.path.join(out_dir, 'log_%d.txt' % idx_process), 'a')
    env = Env(flog=flog, show_gui=(not args.no_gui))
    cam = Camera(env, theta=3.159759861190408, phi=0.7826405702413783)
    if not args.no_gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)



    AIP_old = model_AIP.AIP(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim).to(device).eval()

    if not torch.cuda.is_available():
        device = "cpu"
    if args.primact_type == "pulling":
        train_conf = torch.load(
            os.path.join(
                "../where2actPP/logs/final_logs/finalexp-model_all_final-pulling-None-train_all_v1",
                'conf.pth'))
        actor = model_3d_w2a.Network(feat_dim=train_conf.feat_dim, rv_dim=train_conf.rv_dim,
                                     rv_cnt=train_conf.rv_cnt)
        data_to_restore = torch.load(
            os.path.join(
                "../where2actPP/logs/final_logs/finalexp-model_all_final-pulling-None-train_all_v1",
                'ckpts', '81-network.pth'))
        actor.load_state_dict(data_to_restore, strict=False)
        actor.to(device).eval()
    else:
        train_conf = torch.load(
            os.path.join(
                "../where2actPP/logs/final_logs/finalexp-model_all_final-pushing-None-train_all_v1",
                'conf.pth'))
        actor = model_3d_w2a.Network(feat_dim=train_conf.feat_dim, rv_dim=train_conf.rv_dim,
                                     rv_cnt=train_conf.rv_cnt)
        data_to_restore = torch.load(os.path.join(
            "../where2actPP/logs/final_logs/finalexp-model_all_final-pushing-None-train_all_v1",
            'ckpts', '92-network.pth'))
        actor.load_state_dict(data_to_restore, strict=False)
        actor.to(device).eval()


    AAP_old = model_AAP.network(input_dim=17, pnpp_feat_dim=128, hidden_feat_dim=128,
                                            feat_dim=args.feat_dim, hidden_dim=args.hidden_dim)
    data_to_store = torch.load(
        os.path.join(args.AAP_dir, 'ckpts', args.AAP_epoch))
    AAP_old.load_state_dict(data_to_store)
    AAP_old.to(args.device).eval()

    data_to_store = torch.load(
        os.path.join(args.AIP_dir, 'ckpts', args.AIP_epoch))
    AIP_old.load_state_dict(data_to_store)

    ############################################################################################################
    robot_loaded = 0
    tot_done_epoch = 0
    tot_fail_epoch = 0
    robot_urdf_fn = './robots/panda_gripper.urdf'
    prev_epoch_qsize = 0

    old_out_dir = out_dir
    tot_TP = 0
    tot_TN = 0
    tot_FN = 0
    tot_FP = 0
    tot_succ = 0
    robot_material_dict = {}
    object_material_dict = {}
    for epoch in range(20):
        train_batches = enumerate(train_dataloader, 0)
        for train_batch_ind, batch in train_batches:
            if(random.random() < 0.5):
                continue
            now_epoch_qsize = epoch_Q.qsize()
            #################################### load network ###############################
            if now_epoch_qsize > prev_epoch_qsize:
                AIP_old.load_state_dict(torch.load(os.path.join(out_dir, 'ckpts', '%d_secondAIP-network.pth' % (now_epoch_qsize))))
                AAP_old.load_state_dict(torch.load(os.path.join(out_dir, 'ckpts', '%d_secondAAP-network.pth' % (now_epoch_qsize))))
                prev_epoch_qsize = now_epoch_qsize
            ##################################################################################

            ##################################### init input #########################################
            input_pcs_list = []
            dir_list = []
            f_dir_list = []
            push_dis_list = []
            dis_list = []
            ctpt_list = []
            joint_info_list = []
            start_pos_list = []
            end_pos_list = []
            # ['gripper_direction_world', 'gripper_action_dis', 'gt_motion', 'pcs', 'ctpt', 'joint_info',
            #  'gripper_forward_direction_world', 'gt_labels', 'start_pos', 'end_pos', "shape_id", "mat44", "ctpt_2d",
            #  "cur_qpos", "trial_id", "mu1", "mu2", "mu3", "mu4", "density", "friction"]
            # dir, dis, push_dis, ctpt, joint_info, pcs, start_pos, end_pos, f_dir
            ##########################################################################################
            # y_more_zero = 0
            # y_less_zero = 0
            # for idx in range(32):
            #     interaction_ctpt = batch[data_features.index('ctpt')][idx]
            #     interaction_ctpt_y = interaction_ctpt[1]
            #     interaction_gt_label = batch[data_features.index('gt_labels')][idx]
            #     if interaction_gt_label == 1:
            #         if interaction_ctpt_y > 0:
            #             y_more_zero = 1
            #         if interaction_ctpt_y < 0:
            #             y_less_zero = 1
            # idxx = random.sample(range(16, 32), 3)
            idxx = [random.randint(0,30)]
            idx = idxx[0]
            trial_id = batch[data_features.index('trial_id')][idx]
            shape_id = batch[data_features.index('shape_id')][idx]
            ctpt_2d = batch[data_features.index('ctpt_2d')][idx]
            cur_qpos = batch[data_features.index('cur_qpos')][idx]
            st_pos = batch[data_features.index('start_pos')][idx]
            mu1 = batch[data_features.index('mu1')][idx]
            mu2 = batch[data_features.index('mu2')][idx]
            mu3 = batch[data_features.index('mu3')][idx]
            mu4 = batch[data_features.index('mu4')][idx]
            density = batch[data_features.index('density')][idx]
            friction = batch[data_features.index('friction')][idx]
            rand_pos_seed = batch[data_features.index('rand_pos_seed')][idx]
            # out_dir = os.path.join(old_out_dir, f"{train_batch_ind}_{shape_id}")
            # if not os.path.exists(out_dir):
            #     os.mkdir(out_dir)

            torch.cuda.empty_cache()

            if not mu3 in robot_material_dict:
                robot_material_dict[mu3] = env.get_material(mu3, mu4, 0.01)
            if not mu1 in object_material_dict:
                object_material_dict[mu1] = env.get_material(mu1, mu2, 0.01)

            robot_material = robot_material_dict[mu3]
            object_material = object_material_dict[mu1]
            state = 'random-middle'
            object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
            env.load_object(object_urdf_fn, object_material, state=state, density=density)

            cur_qpos = np.array(cur_qpos)
            env.object.set_qpos(cur_qpos.astype(np.float32))
            ############## get target part ###############
            env.step()
            env.render()
            rgb, depth = cam.get_observation()

            cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
            cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

            object_link_ids = env.movable_link_ids
            gt_movable_link_mask = cam.get_movable_link_mask([env.movable_link_ids])

            target_part_id = object_link_ids[gt_movable_link_mask[ctpt_2d[0], ctpt_2d[1]] - 1]
            # print("target_part_id: ", target_part_id)
            env.set_target_object_part_actor_id(target_part_id)
            target_part_joint_idx = env.target_object_part_joint_id
            gt_movable_link_mask = cam.get_movable_link_mask([target_part_id])
            mat44 = cam.get_metadata()['mat44']
            #############################################################################
            world_pc, pc, movables = generate_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, gt_movable_link_mask, mat44,
                                                 device=args.device)
            #############################################################################
            # ['gripper_direction_world', 'gripper_action_dis', 'gt_motion', 'pcs', 'ctpt', 'joint_info',
            #  'gripper_forward_direction_world', 'gt_labels', 'start_pos', 'end_pos', "shape_id", "mat44", "ctpt_2d",
            #  "cur_qpos", "trial_id", "mu1", "mu2", "mu3", "mu4", "density", "friction"]
            input_pcs_list.append(world_pc)
            dir_list.append(batch[data_features.index('gripper_direction_world')][idx])
            f_dir_list.append(batch[data_features.index('gripper_forward_direction_world')][idx])
            ctpt_list.append(batch[data_features.index('ctpt')][idx])
            push_dis_list.append(batch[data_features.index('gt_motion')][idx])
            dis_list.append(batch[data_features.index('gripper_action_dis')][idx])
            joint_info_list.append(np.array([1, 1, 1, 1]))
            start_pos_list.append(batch[data_features.index('start_pos')][idx])
            end_pos_list.append(batch[data_features.index('end_pos')][idx])

            ######################################################

            ############## set friction ########################
            for j in env.object.get_joints():
                if j.get_child_link().get_id() == env.target_object_part_actor_id:
                    j.set_friction(friction)
            ####################################################
            tot_step = 0
            other_step = 0
            # dir, dis, push_dis, ctpt, joint_info, pcs, start_pos, end_pos, f_dir
            ############################ get several actions ###############################
            while tot_step < args.step:
                tot_step += 1
                ###########################
                start_pos = float(env.get_object_qpos()[target_part_joint_idx])
                ###########################

                ############################ get new hidden_info #########################
                hidden_info = get_hidden_info(input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list,
                                        joint_info_list, start_pos_list, end_pos_list, AAP_old.hidden_encoder, args)
                #######################################################################
                env.render()
                rgb, depth = cam.get_observation()
                cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
                # cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

                init_gt_movable_link_mask = cam.get_movable_link_mask([target_part_id])
                init_pc_world, init_pc_cam, init_mask = generate_pc(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts,
                                                                    init_gt_movable_link_mask, mat44,
                                                                    device=device)
                init_mask = init_mask.reshape(10000)
                pc_world = init_pc_world[0].detach().cpu().numpy()
                pc_cam = init_pc_cam[0].detach().cpu().numpy()
                # get_aff_pic(init_pc_world, init_pc_cam, pc_cam, init_mask, tot_step, hidden_info, actor, AAP_old, mat44,
                #             start_pos, out_dir, train_batch_ind)

                ########################## get ctpt ##############################
                n = 100
                with torch.no_grad():
                    actions = actor.inference_nactor_whole_pc(init_pc_cam, n=n)
                    pred_Rs = actor.actor.bgs(actions.reshape(-1, 3, 2))
                    up = pred_Rs[:, :, 0]
                    forward = pred_Rs[:, :, 1]
                    mat44_tensor = torch.FloatTensor(cam.get_metadata()['mat44'][:3, :3]).to(args.device)
                    # for i in range(10000*n):
                    #     print(up[i].shape)
                    #     print(mat44_tensor.shape)
                    up = torch.mm(mat44_tensor, up.permute(1, 0)).permute(1, 0)
                    forward = torch.mm(mat44_tensor, forward.permute(1, 0)).permute(1, 0)

                    st_pos = torch.FloatTensor(np.array([start_pos])).to(args.device)
                with torch.no_grad():
                    state_joint_origins = torch.from_numpy(np.ones(n * 40000)).float().view(-1, 4).to(args.device)
                    critic_result = AAP_old.critic.inference_critic_score_diff_naction(up, init_pc_world, None,
                                                                                             state_joint_origins,
                                                                                             forward, hidden_info, st_pos, n=n)
                    result = AIP_old.inference_nscore(init_pc_world, hidden_info, up, forward, critic_result, n=n)

                    result = result.view(10000, n, 1).topk(k=3, dim=1)[0].mean(dim=1).view(-1)
                    result = result.detach().cpu().numpy()
                result = result.reshape(10000)
                # ipdb.set_trace()
                result *= init_mask

                accu = 2
                # print(result)
                xs = np.where(result > accu)[0]
                while len(xs) < 50:
                    accu = accu - 0.01
                    xs = np.where(result > accu)[0]
                    # print("len:", len(xs))
                # print("length:", len(xs))

                if accu < 0.01:
                    break
                p_id = random.randint(0, len(xs) - 1)
                # for i in range(len(xs)):
                #     if pc_world[xs[i]][1] > pc_world[xs[p_id]][1]:
                #         p_id = i
                # print("p_id", p_id)
                # print("xs", xs)
                # print("pc_cam", pc_cam)

                position_cam = pc_cam[xs[p_id]]
                # print(position_cam)
                position_cam[0] += 5
                position_cam_xyz1 = np.ones((4), dtype=np.float32)
                position_cam_xyz1[:3] = position_cam
                position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
                position_world = position_world_xyz1[:3]
                position_cam[0] -= 5
                init_pc_world[0][0] = init_pc_world[0][xs[p_id]]
                init_pc_cam[0][0] = init_pc_cam[0][xs[p_id]]
                init_mask = init_mask.reshape(1, 10000, 1)
                init_mask[0][0] = 1
                ############################################################################

                ################################# get action ###############################
                with torch.no_grad():
                    pred_6d = actor.inference_actor(init_pc_cam)[0]  # RV_CNT x 6
                    pred_Rs = actor.actor.bgs(pred_6d.reshape(-1, 3, 2)).detach().cpu().numpy()

                gripper_direction_camera = pred_Rs[:, :, 0]
                gripper_forward_direction_camera = pred_Rs[:, :, 1]
                # print(gripper_forward_direction_camera.shape)
                rvs = gripper_direction_camera.shape[1]
                scores = []
                for j in range(rvs):
                    up = gripper_direction_camera[j]
                    forward = gripper_forward_direction_camera[j]

                    up = cam.get_metadata()['mat44'][:3, :3] @ up
                    forward = cam.get_metadata()['mat44'][:3, :3] @ forward
                    up = torch.FloatTensor(up).view(1, -1).to(args.device)
                    forward = torch.FloatTensor(forward).view(1, -1).to(args.device)
                    joint_info = torch.FloatTensor(np.array([1, 1, 1, 1])).view(1, -1).to(args.device)
                    position_world_tensor = torch.FloatTensor(position_world).view(1, -1).to(args.device)
                    st_pose = torch.FloatTensor(np.array([start_pos])).view(1, -1).to(args.device)
                    with torch.no_grad():
                        # ipdb.set_trace()
                        # print("pc_shape:", init_pc_world.shape)
                        # print("up_shape:", up.shape)
                        up = up.view(1, -1)
                        AAP_score = AAP_old.critic(up, init_pc_world, position_world_tensor, joint_info,
                                                            forward,
                                                            hidden_info, st_pose)
                        score = AIP_old(init_pc_world, hidden_info, up, forward, AAP_score)[0]
                    scores.append(score.item())

                scores = np.array(scores)
                # print("score_shape:", scores.shape)
                accu = 2
                xs = np.where(scores > accu)[0]
                while len(xs) < 1:
                    accu = accu - 0.05
                    xs = np.where(scores > accu)[0]
                # print("xs:", xs)
                ################################## manipulate ######################################
                up = gripper_direction_camera[xs[0]]
                forward = gripper_forward_direction_camera[xs[0]]

                up = cam.get_metadata()['mat44'][:3, :3] @ up
                forward = cam.get_metadata()['mat44'][:3, :3] @ forward

                up = np.array(up, dtype=np.float32)
                forward = np.array(forward, dtype=np.float32)

                left = np.cross(up, forward)
                left /= np.linalg.norm(left)
                forward = np.cross(left, up)
                forward /= np.linalg.norm(forward)

                action_direction_world = up
                torch.cuda.empty_cache()
                if robot_loaded == 0:
                    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in args.primact_type))
                    robot_loaded = 1
                else:
                    robot.load_gripper(robot_urdf_fn, robot_material, open_gripper=('pulling' in args.primact_type))

                rotmat = np.eye(4).astype(np.float32)
                rotmat[:3, 0] = forward
                rotmat[:3, 1] = left
                rotmat[:3, 2] = up

                final_dist = 0.05

                final_rotmat = np.array(rotmat, dtype=np.float32)
                final_rotmat[:3,
                3] = position_world - action_direction_world * final_dist - action_direction_world * 0.1
                if args.primact_type == 'pushing':
                    final_rotmat[:3,
                    3] = position_world + action_direction_world * final_dist - action_direction_world * 0.15
                final_pose = Pose().from_transformation_matrix(final_rotmat)

                start_rotmat = np.array(rotmat, dtype=np.float32)
                start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
                start_pose = Pose().from_transformation_matrix(start_rotmat)

                end_rotmat = np.array(rotmat, dtype=np.float32)
                end_rotmat[:3, 3] = position_world - action_direction_world * 0.1

                robot.robot.set_root_pose(start_pose)
                env.render()

                # activate contact checking
                env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, False)
                target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()

                success = True
                target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
                position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
                succ_images = []
                if args.primact_type == 'pulling':
                    try:
                        init_success = True
                        success_grasp = False
                        # print("try to grasp")
                        # imgs = robot.wait_n_steps(1000, vis_gif=True, vis_gif_interval=200, cam=cam)
                        # succ_images.extend(imgs)
                        try:
                            robot.open_gripper()
                            robot.move_to_target_pose(end_rotmat, 2000)
                            robot.wait_n_steps(2000)
                            robot.close_gripper()
                            robot.wait_n_steps(2000)
                            now_qpos = robot.robot.get_qpos().tolist()
                            finger1_qpos = now_qpos[-1]
                            finger2_qpos = now_qpos[-2]
                            # print(finger1_qpos, finger2_qpos)
                            if finger1_qpos + finger2_qpos > 0.01:
                                success_grasp = True
                        except Exception:
                            init_success = False
                        if not (success_grasp and init_success):
                            # print('grasp_fail')
                            success = False
                        else:
                            try:
                                robot.move_to_target_pose(final_rotmat, 2000)
                                robot.wait_n_steps(2000)
                            except Exception:
                                # print("fail")
                                success = False
                    except Exception:
                        success = False
                else:
                    try:
                        robot.close_gripper()
                        succ_images = []
                        try:
                            robot.move_to_target_pose(final_rotmat, 2000)
                            robot.wait_n_steps(2000)
                        except Exception:
                            # print("fail")
                            ct_error = ct_error + 1
                            success = False
                    except Exception:
                        success = False

                target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
                position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
                touch_position_world_xyz_start = position_world_xyz1[:3]
                touch_position_world_xyz_end = position_world_xyz1_end[:3]
                env.scene.remove_articulation(robot.robot)
                # succ_cnt = succ_cnt + 1
                flag = 0
                if success:
                    final_target_part_qpos = float(env.get_object_qpos()[target_part_joint_idx])
                    gt_motion = abs(final_target_part_qpos - start_pos)

                    if (final_target_part_qpos - start_pos > 0 and rand_pos_seed == 0) or (
                            final_target_part_qpos - start_pos < 0 and rand_pos_seed == 1):
                        flag = 1
                    mov_dir = touch_position_world_xyz_end - touch_position_world_xyz_start
                    mov_dir /= np.linalg.norm(mov_dir)
                    intended_dir = -up
                    intend_motion = intended_dir @ mov_dir
                    if (intend_motion < 0.5 and args.primact_type == 'pulling'):
                        gt_motion = 0
                        final_target_part_qpos = start_pos
                        cur_qpos[target_part_joint_idx] = start_pos
                        env.set_object_joint_angles(cur_qpos)
                else:
                    gt_motion = 0
                    final_target_part_qpos = start_pos
                    cur_qpos[target_part_joint_idx] = start_pos
                    env.set_object_joint_angles(cur_qpos)
                # if len(succ_images) > 0:
                #     imageio.mimsave(os.path.join(out_dir, f'{train_batch_ind}_{tot_step}_operation.gif'), succ_images)


                input_pcs_list.append(init_pc_world)
                dir_list.append(up)
                f_dir_list.append(forward)
                ctpt_list.append(position_world)
                push_dis_list.append(gt_motion)
                dis_list.append(np.array([0.05]))
                joint_info_list.append(np.array([1, 1, 1, 1]))
                start_pos_list.append(start_pos)
                end_pos_list.append(final_target_part_qpos)

            if(len(dir_list) == 0):
                env.scene.remove_articulation(env.object)
                continue
            print("process ",idx_process, "finish step")
            for i in range(len(input_pcs_list)):
                input_pcs_list[i] = input_pcs_list[i].detach().cpu().numpy()
            np_batch = batch
            for i in range(len(batch[data_features.index('pcs')])):
                np_batch[data_features.index('pcs')][i] = np_batch[data_features.index('pcs')][i].detach().cpu().numpy()
            transition_Q.put(
                [input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list,
                 joint_info_list, start_pos_list, end_pos_list, np_batch])

            env.scene.remove_articulation(env.object)
            torch.cuda.empty_cache()



trans_q = mp.Queue()
epoch_q = mp.Queue()
# args.device = "cuda:1"
for idx_process in range(1):
    p = mp.Process(target=run_an_collect, args=(idx_process, args, trans_q, epoch_q))
    p.start()


AAP = model_AAP.network(input_dim=17, pnpp_feat_dim=128, hidden_feat_dim=128, feat_dim=args.feat_dim,
                                hidden_dim=args.hidden_dim)

AAP_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, AAP.parameters()), lr=args.lr,
                               weight_decay=args.weight_decay)
# learning rate scheduler
AAP_lr_scheduler = torch.optim.lr_scheduler.StepLR(AAP_opt, step_size=args.lr_decay_every,
                                                       gamma=args.lr_decay_by)
utils.optimizer_to_device(AAP_opt, args.device)

data_to_store = torch.load(
    os.path.join(args.AAP_dir, 'ckpts', args.AAP_epoch))
AAP.load_state_dict(data_to_store)
AAP.to(args.device)

AIP = model_AIP.AIP(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim).to(args.device)
data_to_store = torch.load(
        os.path.join(args.AIP_dir, 'ckpts', args.AIP_epoch))
AIP.load_state_dict(data_to_store)
AIP_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, AIP.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)
# learning rate scheduler
AIP_lr_scheduler = torch.optim.lr_scheduler.StepLR(AIP_opt, step_size=args.lr_decay_every,
                                                         gamma=args.lr_decay_by)
utils.optimizer_to_device(AIP_opt, args.device)

train_AAP = True
train_AIP = False
epoch = 1
loss_cnt = 0
while True:
    if not trans_q.empty():

        t0 = time.time()
        print("time:", t0, " update network in epoch:", epoch + 1)
        epoch += 1

        input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list, joint_info_list, start_pos_list, end_pos_list, batch = trans_q.get()
        for i in range(len(input_pcs_list)):
            input_pcs_list[i] = torch.from_numpy(input_pcs_list[i])
        for i in range(len(batch[data_features.index('pcs')])):
            batch[data_features.index('pcs')][i] = torch.from_numpy(batch[data_features.index('pcs')][i])
        if train_AAP:
            hidden_info = get_hidden_info(input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list,
                                    joint_info_list, start_pos_list, end_pos_list, AAP.hidden_encoder, args,
                                    grad=True)
            loss = critic_forward(batch, data_features, AAP, args, hidden_info)
            AAP_opt.zero_grad()
            loss.backward()
            # optimize one step
            AAP_opt.step()
            AAP_opt.zero_grad()
            AAP_lr_scheduler.step()
        if train_AIP and len(dir_list) > 1:
            loss = AIP_forward(batch, data_features, AIP, AAP, args, input_pcs_list, dir_list, f_dir_list,
                               push_dis_list, dis_list, ctpt_list, joint_info_list, start_pos_list, end_pos_list)
            AIP_opt.zero_grad()
            loss.backward()
            # optimize one step
            AIP_opt.step()
            AIP_opt.zero_grad()
            AIP_lr_scheduler.step()
        if train_AAP or (train_AIP and len(dir_list) > 1):
            loss_cnt += loss.item()

        if epoch % 10 == 0:
            print("epoch: ", epoch, "loss: ", loss_cnt / 10)
            loss_cnt = 0

        if epoch % 400 == 0:
            step = epoch // 400
            with torch.no_grad():

                torch.save(AAP.state_dict(), os.path.join(out_dir, 'ckpts', f'{step}_secondAAP-network.pth'))
                torch.save(AIP.state_dict(), os.path.join(out_dir, 'ckpts', f'{step}_secondAIP-network.pth'))
            epoch_q.put(step)

        if epoch % 1200 == 0:
            if train_AAP:
                train_AAP = False
            else :
                train_AAP = True

            if train_AIP:
                train_AIP = False
            else:
                train_AIP = True