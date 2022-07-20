"""
    SAPIENVisionDataset
        Joint data loader for six primacts
        for panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
"""

import os
import h5py
import torch
import torch.utils.data as data
import numpy as np
import json
# from pyquaternion import Quaternion
from camera import Camera


class SAPIENVisionDataset(data.Dataset):

    def __init__(self, primact_types, data_features):
        self.primact_types = primact_types

        # data features
        self.data_features = data_features

        self.data_buffer = []  # (gripper_direction_world, gripper_action_dis, gt_motion)
        self.seq = []

    def get_seq(self):
        length = len(self.data_buffer) // 32
        self.seq = np.arange(length)
        np.random.shuffle(self.seq)

    def load_data(self, dir, batch_size=32, ignore_joint_info=True):

        for i in range(40):
            print(f"process{i} : \n")
            cur_dir = os.path.join(dir, f'process_{i}')
            for j in range(30):
                if not os.path.exists(os.path.join(cur_dir, f'result_{j}_{batch_size}.json')):
                    continue

                for k in range(1,batch_size+1):
                    with open(os.path.join(cur_dir, f'result_{j}_{k}.json'), 'r') as fin:
                        result_data = json.load(fin)
                        rand_pos_seed = None
                        cur_joint_dis = None

                        gt_labels = result_data['gt_labels']
                        phy_info = None

                        gripper_direction_world = np.array(result_data['gripper_direction_world'], dtype=np.float32)
                        gripper_forward_direction_world = np.array(result_data['gripper_forward_direction_world'], dtype=np.float32)
                        gripper_action_dis = result_data['dist']

                        gt_motion = result_data['start_target_part_qpos'] - result_data['final_target_part_qpos']

                        mu1 = result_data['mu1']
                        mu2 = result_data['mu2']
                        mu3 = result_data['mu3']
                        mu4 = result_data['mu4']
                        density = result_data['density']
                        pc_dir = os.path.join(cur_dir,f'cam_XYZA_{j}_{k}.h5')
                        camera_metadata = result_data['camera_metadata']
                        # cam_theta, cam_phi = camera_metadata['theta'], camera_metadata['phi']
                        mat44 = np.array(camera_metadata['mat44'], dtype=np.float32)
                        ctpt = result_data['position_world']
                        joint_info = result_data['joint_info']
                        friction = result_data['friction']
                        if ignore_joint_info:
                            joint_info = [0,0,0,0]
                        ori_pixel_ids = np.array(result_data['pixel_locs'], dtype=np.int32)
                        start_pos = result_data['start_target_part_qpos']
                        end_pos = result_data['final_target_part_qpos']

                        ctpt_2d = result_data['pixel_locs']
                        shape_id = result_data['shape_id']
                        cur_qpos = result_data['cur_qpos']
                        self.data_buffer.append((gripper_direction_world, gripper_action_dis, gt_motion, mu1, density,
                                                 pc_dir, mat44, ctpt, joint_info, ori_pixel_ids, friction, start_pos,
                                                 end_pos, shape_id, ctpt_2d, cur_qpos, gripper_forward_direction_world, mu2, mu3, mu4, gt_labels, phy_info, j, i, rand_pos_seed,cur_joint_dis))

    def __str__(self):
        return "PhysicsDataLoader"

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, index):
        ind = index // 32
        index = self.seq[ind] * 32 + (index % 32)
        gripper_direction_world, gripper_action_dis, gt_motion, mu1, density, pc_dir, mat44, ctpt, joint_info, ori_pixel_ids, friction, start_pos, end_pos, shape_id, ctpt_2d, cur_qpos, gripper_forward_direction_world, mu2, mu3, mu4, gt_labels, phy_info, trial_id, pro_id, rand_pos_seed,cur_joint_dis = \
            self.data_buffer[index]


        # print(result, is_original)
        data_feats = ()
        # out2 = None
        # out3 = None
        for feat in self.data_features:

            if feat == 'gripper_direction_world':
                out = gripper_direction_world
                data_feats = data_feats + (out,)
            elif feat == 'gripper_direction_cam':
                out = np.linalg.inv(mat44[:3, :3]) @ gripper_direction_world
                data_feats = data_feats + (out,)
            elif feat == 'gt_motion':
                out = gt_motion
                data_feats = data_feats + (out,)

            elif feat == 'gripper_action_dis':
                out = gripper_action_dis
                data_feats = data_feats + (out,)

            elif feat == 'mu1':
                out = mu1
                data_feats = data_feats + (out,)

            elif feat == 'density':
                out = density
                data_feats = data_feats + (out,)

            # elif feat == 'pc_dir':
            #     out = pc_dir
            #     data_feats = data_feats + (out,)

            elif feat == 'ctpt':
                out = ctpt
                data_feats = data_feats + (out,)
            elif feat == 'ctpt_cam':
                x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                with h5py.File(os.path.join(pc_dir), 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
                pt = out[x, y, :3]
                data_feats = data_feats + (pt,)
            elif feat == 'joint_info':
                out = joint_info
                data_feats = data_feats + (out,)

            elif feat == 'trial_id':
                out = trial_id
                data_feats = data_feats + (out,)

            elif feat == 'pcs':
                x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                with h5py.File(os.path.join(pc_dir), 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
                # with Image.open(os.path.join(cur_dir, 'interaction_mask_%d.png' % result_idx)) as fimg:
                #     out3 = (np.array(fimg, dtype=np.float32) > 127)
                pt = out[x, y, :3]
                # ptid = np.array([x, y], dtype=np.int32)
                mask = (out[:, :, 3] > 0.5)
                mask[x, y] = False
                pc = out[mask, :3]
                # pcids = grid_xy[:, mask].T
                # out3 = out3[mask]
                idx = np.arange(pc.shape[0])
                np.random.shuffle(idx)
                while len(idx) < 30000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:30000-1]
                pc = pc[idx, :]
                # pcids = pcids[idx, :]
                # out3 = out3[idx]
                pc = np.vstack([pt, pc])
                # pcids = np.vstack([ptid, pcids])
                # out3 = np.append(True, out3)
                # normalize to zero-centered
                pc[:, 0] -= 5
                pc = (mat44[:3, :3] @ pc.T).T
                pc = np.array(pc, dtype=np.float32)

                out = torch.from_numpy(pc).unsqueeze(0)
                # out2 = torch.from_numpy(pcids).unsqueeze(0)
                # out3 = torch.from_numpy(out3).unsqueeze(0).float()
                data_feats = data_feats + (out,)
            elif feat == 'pcs_cam':
                x, y = ori_pixel_ids[0], ori_pixel_ids[1]
                with h5py.File(os.path.join(pc_dir), 'r') as fin:
                    cam_XYZA_id1 = fin['id1'][:].astype(np.int64)
                    cam_XYZA_id2 = fin['id2'][:].astype(np.int64)
                    cam_XYZA_pts = fin['pc'][:].astype(np.float32)
                out = Camera.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, 448, 448)
                # with Image.open(os.path.join(cur_dir, 'interaction_mask_%d.png' % result_idx)) as fimg:
                #     out3 = (np.array(fimg, dtype=np.float32) > 127)
                pt = out[x, y, :3]
                # ptid = np.array([x, y], dtype=np.int32)
                mask = (out[:, :, 3] > 0.5)
                mask[x, y] = False
                pc = out[mask, :3]
                # pcids = grid_xy[:, mask].T
                # out3 = out3[mask]
                idx = np.arange(pc.shape[0])
                np.random.shuffle(idx)
                while len(idx) < 30000:
                    idx = np.concatenate([idx, idx])
                idx = idx[:30000-1]
                pc = pc[idx, :]
                # pcids = pcids[idx, :]
                # out3 = out3[idx]
                pc = np.vstack([pt, pc])
                # pcids = np.vstack([ptid, pcids])
                # out3 = np.append(True, out3)
                # normalize to zero-centered
                pc[:, 0] -= 5
                pc = np.array(pc, dtype=np.float32)

                out = torch.from_numpy(pc).unsqueeze(0)
                data_feats = data_feats + (out,)
            elif feat == 'friction':
                out = friction
                data_feats = data_feats + (out,)
            elif feat == 'start_pos':
                out = start_pos
                data_feats = data_feats + (out,)
            elif feat == 'end_pos':
                out = end_pos
                data_feats = data_feats + (out,)
            elif feat == 'shape_id':
                out = shape_id
                data_feats = data_feats + (out,)
            elif feat == 'ctpt_2d':
                out = ctpt_2d
                data_feats = data_feats + (out,)
            elif feat == 'cur_qpos':
                out = cur_qpos
                data_feats = data_feats + (out,)
            elif feat == 'gripper_forward_direction_world':
                out = gripper_forward_direction_world
                data_feats = data_feats + (out,)
            elif feat == 'gripper_forward_direction_cam':
                out = np.linalg.inv(mat44[:3, :3]) @ gripper_forward_direction_world
                data_feats = data_feats + (out,)
            elif feat == 'mat44':
                out = mat44
                data_feats = data_feats + (out,)
            elif feat == 'mu2':
                out = mu2
                data_feats = data_feats + (out,)
            elif feat == 'mu3':
                out = mu3
                data_feats = data_feats + (out,)
            elif feat == 'mu4':
                out = mu4
                data_feats = data_feats + (out,)
            elif feat == 'gt_labels':
                out = gt_labels
                data_feats = data_feats + (out,)
            elif feat == 'phy_info':
                out = phy_info
                data_feats = data_feats + (out,)
            elif feat == 'pro_id':
                out = pro_id
                data_feats = data_feats + (out,)
            elif feat == 'rand_pos_seed':
                out = rand_pos_seed
                data_feats = data_feats + (out,)
            elif feat == 'cur_joint_dis':
                out = cur_joint_dis
                data_feats = data_feats + (out,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

