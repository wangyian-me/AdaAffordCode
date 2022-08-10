"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
from utils import get_global_position_from_camera, save_h5, radian2degree, degree2radian
# import cv2
import json
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
from models import model_AAP, model_AIP, model_3d_w2a, model_AAP_affordance, model_AIP_affordance
import imageio
import ipdb
parser = ArgumentParser()

parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--num_point_per_shape', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--primact_type', type=str, default='pushing', help='the primact type')
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--AAP_dir', type=str, default='nothing')
parser.add_argument('--AAP_epoch', type=str, default='nothing')
parser.add_argument('--AIP_dir', type=str, default='nothing')
parser.add_argument('--AIP_epoch', type=str, default='nothing')
parser.add_argument('--AIP_aff_dir', type=str, default='nothing')
parser.add_argument('--AIP_aff_epoch', type=str, default='nothing')
parser.add_argument('--AAP_aff_dir', type=str, default='nothing')
parser.add_argument('--AAP_aff_epoch', type=str, default='nothing')
parser.add_argument('--out_dir', type=str, default='../logs')
parser.add_argument('--step', type=int, default=4)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--try_per_batch', type=int, default=10)
args = parser.parse_args()

data_features = ['gripper_direction_world', 'gripper_action_dis', 'gt_motion', 'pcs', 'ctpt', 'joint_info',
                     'gripper_forward_direction_world', 'gt_labels', 'start_pos', 'end_pos', "shape_id", "mat44", "ctpt_2d", "cur_qpos", "trial_id","mu1","mu2","mu3","mu4","density","friction"]


def get_our_pred(batch, data_features, network, conf, hidden_info):
    dir = batch[data_features.index('gripper_direction_world')]
    input_pcs = batch[data_features.index('pcs')]
    ctpt = batch[data_features.index('ctpt')]
    joint_info = batch[data_features.index('joint_info')]
    f_dir = batch[data_features.index('gripper_forward_direction_world')]
    gt_labels = batch[data_features.index('gt_labels')]
    # hidden_info = batch[data_features.index('hidden_info')]

    dir = torch.FloatTensor(np.array(dir)).to(conf.device)
    batch_size = dir.shape[0]
    dir = dir.view(batch_size, -1)
    hidden_info = hidden_info.repeat(batch_size, 1)
    # hidden_info = torch.FloatTensor(np.array(hidden_info)).view(batch_size, -1).to(conf.device)
    ctpt = torch.FloatTensor(np.array(ctpt)).view(batch_size, -1).to(conf.device)
    joint_info = torch.FloatTensor(np.array(joint_info)).view(batch_size, -1).to(conf.device)
    f_dir = torch.FloatTensor(np.array(f_dir)).view(batch_size, -1).to(conf.device)
    # gt_labels = torch.FloatTensor(np.array(gt_labels)).view(batch_size).to(conf.device)
    input_pcs = torch.cat(input_pcs, dim=0).to(conf.device)  # B x 3N x 3   # point cloud
    # input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)  # B x 3N x 2
    batch_size = input_pcs.shape[0]

    # print(ctpt.shape)
    # print(f_dir.shape)
    # print(joint_info.shape)
    # print(hidden_info.shape)
    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(
        -1)  # BN
    input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
    ctpt = ctpt.view(batch_size, -1).to(conf.device)
    input_pcs[:, 0] = ctpt

    # pred_score = network(pcs,query_feats,hidden_info_ctpt)

    start_pos = batch[data_features.index('start_pos')]
    start_pos = torch.FloatTensor(np.array(start_pos)).view(batch_size, -1).to(conf.device)
    with torch.no_grad():
        pred = network.critic(dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, start_pos)  # B x 2, B x F x N

    return torch.sigmoid(pred)

def get_hidden_info(input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list, joint_info_list, start_pos_list, end_pos_list, Encoder, conf, grad=False):
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
    if grad == True:
        hidden_info = Encoder(dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir)
    else :
        with torch.no_grad():
            hidden_info = Encoder(dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir)

    return hidden_info

# load dataset

################################# load data ###################################
train_dataset = SAPIENVisionDataset([args.primact_type], data_features)
conf = args
conf.ignore_joint_info = True
conf.continuous = False
train_dataset.load_data(conf.data_dir, batch_size=conf.batch_size)
length = len(train_dataset.data_buffer) // 32
train_dataset.seq = np.arange(length)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                   pin_memory=True, \
                                                   num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                                   worker_init_fn=utils.worker_init_fn)
train_batches = enumerate(train_dataloader, 0)

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
    world_pc = torch.from_numpy(world_pc).float().unsqueeze(0).to(device)
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


def run_an_collect(idx_process, args):

    np.random.seed(random.randint(1, 1000) + idx_process)
    random.seed(random.randint(1, 1000) + idx_process + 7)

    device = args.device

    out_dir = os.path.join(args.out_dir, args.exp_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    # setup env
    flog = open(os.path.join(out_dir, 'log.txt'), 'w')
    env = Env(flog=flog, show_gui=(not args.no_gui))
    cam = Camera(env, theta=3.159759861190408, phi=0.7826405702413783)
    if not args.no_gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

    AIP_map = model_AIP_affordance.Affordance(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim).to(device)
    AIP = model_AIP.AIP(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim).to(device)

    if args.primact_type == "pulling":
        train_conf = torch.load(
            os.path.join("../where2actPP/logs/final_logs/finalexp-model_all_final-pulling-None-train_all_v1",
                         'conf.pth'))
        actor = model_3d_w2a.Network(feat_dim=train_conf.feat_dim, rv_dim=train_conf.rv_dim, rv_cnt=train_conf.rv_cnt)
        data_to_restore = torch.load(
            os.path.join("../where2actPP/logs/final_logs/finalexp-model_all_final-pulling-None-train_all_v1",
                         'ckpts', '81-network.pth'))
        actor.load_state_dict(data_to_restore, strict=False)
        actor.to(device).eval()
    else :
        train_conf = torch.load(
            os.path.join("../where2actPP/logs/final_logs/finalexp-model_all_final-pushing-None-train_all_v1",
                         'conf.pth'))
        actor = model_3d_w2a.Network(feat_dim = train_conf.feat_dim, rv_dim = train_conf.rv_dim, rv_cnt = train_conf.rv_cnt)
        data_to_restore = torch.load(os.path.join("../where2actPP/logs/final_logs/finalexp-model_all_final-pushing-None-train_all_v1", 'ckpts', '92-network.pth'))
        actor.load_state_dict(data_to_restore, strict=False)
        actor.to(device).eval()

    aff_map = model_AAP_affordance.Affordance(feat_dim=args.feat_dim, hidden_dim=args.hidden_dim).to(device)
    data_to_store = torch.load(
        os.path.join(args.AAP_aff_dir, 'ckpts', args.AAP_aff_epoch))
    aff_map.load_state_dict(data_to_store)

    AAP_old = model_AAP.network(input_dim=17, pnpp_feat_dim=128, hidden_feat_dim=128, feat_dim=args.feat_dim, hidden_dim=args.hidden_dim)
    data_to_store = torch.load(
        os.path.join(args.AAP_dir, 'ckpts', args.AAP_epoch))
    AAP_old.load_state_dict(data_to_store)
    AAP_old.to(args.device)

    data_to_store = torch.load(
        os.path.join(args.AIP_dir, 'ckpts', args.AIP_epoch))
    AIP.load_state_dict(data_to_store)

    data_to_store = torch.load(
        os.path.join(args.AIP_aff_dir, 'ckpts', args.AIP_aff_epoch))
    AIP_map.load_state_dict(data_to_store)
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
    for train_batch_ind, batch in train_batches:
        # now_epoch_qsize = epoch_Q.qsize()

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

        # idxx = random.sample(range(16, 32), 3)
        idxx = [0]
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

        out_dir = os.path.join(old_out_dir, f"{train_batch_ind}_{shape_id}")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        torch.cuda.empty_cache()
        robot_material = env.get_material(mu3, mu4, 0.01)
        object_material = env.get_material(mu1, mu2, 0.01)
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
        print("target_part_id: ", target_part_id)
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

            with torch.no_grad():
                aff_score = aff_map.inference_whole_pc(init_pc_world, hidden_info)
                result = AIP_map.inference_whole_pc(init_pc_world, hidden_info, aff_score)
                result = result.detach().cpu().numpy()
            result = result.reshape(10000)
            # ipdb.set_trace()
            result *= init_mask
            # utils.render_pts_label_png(fn, pc_cam, result)
            accu = 2
            # print(result)
            xs = np.where(result > accu)[0]
            while len(xs) < 100:
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
            ############################################ get w2a action ####################################
            with torch.no_grad():
                pred_6d = actor.inference_actor(init_pc_cam)[0]  # RV_CNT x 6
                pred_Rs = actor.actor.bgs(pred_6d.reshape(-1, 3, 2)).detach().cpu().numpy()

            gripper_direction_camera = pred_Rs[:, :, 0]
            gripper_forward_direction_camera = pred_Rs[:, :, 1]
            print(gripper_forward_direction_camera.shape)
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
                    print("pc_shape:", init_pc_world.shape)
                    print("up_shape:", up.shape)
                    up = up.view(1, -1)
                    critic_score = AAP_old.critic(up, init_pc_world, position_world_tensor, joint_info, forward,
                                                        hidden_info, st_pose)
                    score = AIP(init_pc_world, hidden_info, up, forward, critic_score)[0]
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

            if robot_loaded == 0:
                robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in args.primact_type))
                robot_loaded = 1
            else:
                robot.load_gripper(robot_urdf_fn, robot_material, open_gripper=('pulling' in args.primact_type))

            # final_dist = 0.3 + np.random.rand() * 0.25 + trial_id * 0.05
            rotmat = np.eye(4).astype(np.float32)
            rotmat[:3, 0] = forward
            rotmat[:3, 1] = left
            rotmat[:3, 2] = up

            # final_dist = 0.3 + np.random.rand() * 0.25 + trial_id * 0.05
            final_dist = 0.05
            print(final_dist)

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
            env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in args.primact_type)
            target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()

            success = True
            target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
            position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
            succ_images = []
            if args.primact_type == 'pulling':
                try:
                    init_success = True
                    success_grasp = False
                    print("try to grasp")
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
                        print('grasp_fail')
                        success = False
                    else:
                        try:
                            robot.move_to_target_pose(final_rotmat, 2000)
                            robot.wait_n_steps(2000)
                        except Exception:
                            print("fail")
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
                        print("fail")
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
            if success:
                final_target_part_qpos = float(env.get_object_qpos()[target_part_joint_idx])
                gt_motion = abs(final_target_part_qpos - start_pos)

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


        hidden_info = get_hidden_info(input_pcs_list, dir_list, f_dir_list, push_dis_list, dis_list, ctpt_list,
                                joint_info_list, start_pos_list, end_pos_list, AAP_old.hidden_encoder, args)
        ############### scoring ###############
        our_pred = get_our_pred(batch, data_features, AAP_old, args, hidden_info).detach().cpu().numpy()
        critic_threshold = 0.5
        pred = []
        for idx in range(our_pred.shape[0]):
            if our_pred[idx] > critic_threshold:
                pred.append(1)
            else:
                pred.append(0)
        gt_result = np.array(batch[data_features.index('gt_labels')])
        for i in range(gt_result.shape[0]):
            if gt_result[i] < 1:
                gt_result[i] = 0
            else:
                gt_result[1] = 1
        TP, TN, FN, FP = utils.cal_score(np.array(pred), gt_result)
        tot_TP += TP
        tot_TN += TN
        tot_FN += FN
        tot_FP += FP

        if tot_TP + tot_FP == 0:
            p = 0
        else:
            p = tot_TP / (tot_TP + tot_FP)
        if tot_TP + tot_FN == 0:
            r = 0
        else:
            r = tot_TP / (tot_TP + tot_FN)
        if r + p == 0:
            F1 = 0
        else:
            F1 = 2 * r * p / (r + p)

        print(tot_TP, tot_TN, tot_FN, tot_FP)
        print(f"ours Fscore : {F1} precision : {p} recall : {r} accu : {(tot_TP + tot_TN) / (tot_TP + tot_TN + tot_FP + tot_FN)}")
        #####################################
        for i in range(args.try_per_batch):
            cur_qpos = batch[data_features.index('cur_qpos')][idx]
            st_pos = batch[data_features.index('start_pos')][idx]
            env.set_object_joint_angles(cur_qpos)
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
            with torch.no_grad():
                result = aff_map.inference_whole_pc(init_pc_world, hidden_info)
                result = result.detach().cpu().numpy()
            result = result.reshape(10000)
            # ipdb.set_trace()
            result *= init_mask
            accu = 0.95
            # print(result)
            xs = np.where(result > accu)[0]
            while len(xs) < 100:
                accu = accu - 0.03
                xs = np.where(result > accu)[0]
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
            ############################################ get w2a action ####################################
            with torch.no_grad():
                pred_6d = actor.inference_actor(init_pc_cam)[0]  # RV_CNT x 6
                pred_Rs = actor.actor.bgs(pred_6d.reshape(-1, 3, 2)).detach().cpu().numpy()

            gripper_direction_camera = pred_Rs[:, :, 0]
            gripper_forward_direction_camera = pred_Rs[:, :, 1]
            print(gripper_forward_direction_camera.shape)
            rvs = gripper_direction_camera.shape[1]
            scores = []
            for j in range(rvs):
                up = gripper_direction_camera[j]
                forward = gripper_forward_direction_camera[j]

                up = cam.get_metadata()['mat44'][:3, :3] @ up
                forward = cam.get_metadata()['mat44'][:3, :3] @ forward
                up = torch.FloatTensor(up).view(1, -1).to(args.device)
                forward = torch.FloatTensor(forward).view(1, -1).to(args.device)
                joint_info = torch.FloatTensor(np.array([1,1,1,1])).view(1, -1).to(args.device)
                position_world_tensor = torch.FloatTensor(position_world).view(1, -1).to(args.device)
                st_pose = torch.FloatTensor(np.array([start_pos])).view(1, -1).to(args.device)
                with torch.no_grad():
                    # ipdb.set_trace()
                    print("pc_shape:", init_pc_world.shape)
                    print("up_shape:", up.shape)
                    up = up.view(1, -1)
                    critic_score = AAP_old.critic(up, init_pc_world, position_world_tensor, joint_info, forward, hidden_info, st_pose)
                scores.append(critic_score.item())

            scores = np.array(scores)
            # print("score_shape:", scores.shape)
            accu = 0.95
            xs = np.where(scores > accu)[0]
            while len(xs) < 3:
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

            if robot_loaded == 0:
                robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in args.primact_type))
                robot_loaded = 1
            else:
                robot.load_gripper(robot_urdf_fn, robot_material, open_gripper=('pulling' in args.primact_type))

            # final_dist = 0.3 + np.random.rand() * 0.25 + trial_id * 0.05
            rotmat = np.eye(4).astype(np.float32)
            rotmat[:3, 0] = forward
            rotmat[:3, 1] = left
            rotmat[:3, 2] = up

            # final_dist = 0.3 + np.random.rand() * 0.25 + trial_id * 0.05
            final_dist = 0.05
            print(final_dist)

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
            env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in args.primact_type)
            target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()

            success = True
            target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
            position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
            succ_images = []
            if args.primact_type == 'pulling':
                try:
                    init_success = True
                    success_grasp = False
                    print("try to grasp")
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
                        print('grasp_fail')
                        success = False
                    else:
                        try:
                            robot.move_to_target_pose(final_rotmat, 2000)
                            robot.wait_n_steps(2000)
                        except Exception:
                            print("fail")
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
                        print("fail")
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
            if success:
                final_target_part_qpos = float(env.get_object_qpos()[target_part_joint_idx])
                gt_motion = abs(final_target_part_qpos - st_pos)

                mov_dir = touch_position_world_xyz_end - touch_position_world_xyz_start
                mov_dir /= np.linalg.norm(mov_dir)
                intended_dir = -up
                intend_motion = intended_dir @ mov_dir
                if (args.primact_type == 'pushing' or intend_motion > 0.5) and gt_motion > 0.01:
                    tot_succ += 1

        print("our_succ", tot_succ / ((train_batch_ind+1)*args.try_per_batch))
        print(
            f"ours Fscore : {F1} precision : {p} recall : {r} accu : {(tot_TP + tot_TN) / (tot_TP + tot_TN + tot_FP + tot_FN)}")
        env.scene.remove_articulation(env.object)

run_an_collect(1,args)