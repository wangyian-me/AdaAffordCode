"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import shutil
import numpy as np
from utils import get_global_position_from_camera, save_h5
import json
from argparse import ArgumentParser
from sapien.core import Pose, ArticulationJointType, ActorDynamicBase
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot

import random
import copy


parser = ArgumentParser()
parser.add_argument('--primact_type', type=str, default='pulling')
parser.add_argument('--out_dir', type=str, default='../data')
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--num_processes', type=int, default=40)
parser.add_argument('--date', type=str, default='0000')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--final_dist', type=float, default=0.05)
parser.add_argument('--min_mass', type=float, default=0.01)
parser.add_argument('--max_mass', type=float, default=0.05)
parser.add_argument('--true_thres', type=float, default=0.01)
parser.add_argument('--collect_num', type=int, default=1)
args = parser.parse_args()


def run_collect(cnt_id=args.cnt_id, trial_id=args.trial_id, primact_type=args.primact_type,
                out_dir=args.out_dir):
    if args.test:
        out_dir = os.path.join(out_dir, '%s_pull_val_%s' % (args.date, args.collect_num))
    else:
        out_dir = os.path.join(out_dir, '%s_pull__%s' % (args.date, args.collect_num))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, 'process_%d' % (cnt_id))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_info = dict()
    np.random.seed(random.randint(1, 1000) + cnt_id)
    random.seed(random.randint(1, 1000) + cnt_id)
    train_file_dir = "../stats/train_10cats_train_data_list.txt"
    if args.test:
        train_file_dir = "../stats/train_10cats_test_data_list.txt"
    all_shape_list = []
    train_shape_list = []
    all_cat_list = ['StorageFurniture', 'Kettle', 'Box', 'Refrigerator', 'Switch', 'TrashCan', 'Window', 'Faucet',
                    'Microwave', 'Door']
    # all_cat_list = ['StorageFurniture', 'Microwave', 'Refrigerator', 'WashingMachine']
    tot_cat = len(all_cat_list)
    len_shape = {}
    len_train_shape = {}
    shape_cat_dict = {}
    cat_shape_id_list = {}
    train_cat_shape_id_list = {}
    for cat in all_cat_list:
        len_shape[cat] = 0
        len_train_shape[cat] = 0
        cat_shape_id_list[cat] = []
        train_cat_shape_id_list[cat] = []

    with open(train_file_dir, 'r') as fin:
        for l in fin.readlines():
            shape_id, cat = l.rstrip().split()
            if cat not in all_cat_list:
                continue
            train_shape_list.append(shape_id)
            all_shape_list.append(shape_id)
            shape_cat_dict[shape_id] = cat
            len_shape[cat] += 1
            len_train_shape[cat] += 1
            cat_shape_id_list[cat].append(shape_id)
            train_cat_shape_id_list[cat].append(shape_id)

    len_train_shape_list = len(train_shape_list)
    len_all_shape_list = len(all_shape_list)
    print(len_shape)

    flog = open(os.path.join(out_dir, 'log.txt'), 'w')
    env = Env(flog=flog, show_gui=(not args.no_gui))

    cam = Camera(env, theta=3.159759861190408, phi=0.7826405702413783)
    out_info['camera_metadata'] = cam.get_metadata_json()
    if not args.no_gui:
        env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)

    random_range = 200
    density_range = 30
    mu1 = 4
    mu2 = 4
    mu3 = 4
    mu4 = 4
    density = 1
    friction = 0
    target_part_id = 0
    while True:
        selected_cat = all_cat_list[random.randint(0, len(all_cat_list) - 1)]
        shape_id = train_cat_shape_id_list[selected_cat][random.randint(0, len_train_shape[selected_cat] - 1)]
        out_info['shape_id'] = shape_id
        object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
        flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
        object_material = env.get_material(mu1, mu2, 0.01)
        state = 'random-closed-middle'
        if np.random.random() < 0.5:
            state = 'closed'
        env.load_object(object_urdf_fn, object_material, state=state, density=density)
        # print("LOADED shape")
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper

        cur_qpos = env.get_object_qpos()
        # simulate some steps for the object to stay rest
        still_timesteps = 0
        wait_timesteps = 0
        while still_timesteps < 150 and wait_timesteps < 200:
            env.step()
            env.render()
            cur_new_qpos = env.get_object_qpos()
            invalid_contact = False
            for c in env.scene.get_contacts():
                for p in c.points:
                    if abs(p.impulse @ p.impulse) > 1e-4:
                        invalid_contact = True
                        break
                if invalid_contact:
                    break
            if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
                still_timesteps += 1
            else:
                still_timesteps = 0
            cur_qpos = cur_new_qpos
            wait_timesteps += 1
        if still_timesteps < 150:
            env.scene.remove_articulation(env.object)
            print("not still")
            continue
        out_info['cur_qpos'] = cur_qpos.tolist()
        cam.get_observation()

        object_link_ids = env.movable_link_ids
        # print(env.movable_link_ids)
        gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
        xs, ys = np.where(gt_movable_link_mask > 0)
        if len(xs) == 0:
            env.scene.remove_articulation(env.object)
            print("cant find ctpt")
            continue

        idx = np.random.randint(len(xs))
        x, y = xs[idx], ys[idx]
        target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
        env.set_target_object_part_actor_id(target_part_id)
        tot_trial = 0
        break
    for i in env.object.get_joints():
        if i.get_child_link().get_id() == target_part_id:
            mass = i.get_child_link().get_mass()
            min_density = args.min_mass * (density / mass)
            max_density = args.max_mass * (density / mass)

    env.scene.remove_articulation(env.object)
    robot_urdf_fn = './robots/panda_gripper.urdf'
    part_x, part_y = x, y
    ori_cur_qpos = copy.deepcopy(cur_qpos)
    while trial_id < 10:
        density = min_density + (max_density - min_density) * random.random()
        mu = random.random() * 20
        mu1 = mu
        mu2 = mu
        mu3 = mu
        mu4 = mu
        friction = random.random() * 300

        succ_cnt = 0

        print("succ_cnt : ",succ_cnt)
        if succ_cnt >= 32:
            trial_id += 1
            continue
        robot_material = env.get_material(mu3, mu4, 0.01)
        object_material = env.get_material(mu1, mu2, 0.01)
        state = 'random-middle'
        env.load_object(object_urdf_fn, object_material, state=state, density=density)
        # density = density + 6
        env.object.set_qpos(ori_cur_qpos.astype(np.float32))
        # print("LOADED shape")
        out_info['joint_angles_lower'] = env.joint_angles_lower
        out_info['joint_angles_upper'] = env.joint_angles_upper

        # simulate some steps for the object to stay rest
        env.step()
        env.render()

        rgb, depth = cam.get_observation()
        # Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

        cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
        cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
        save_h5(os.path.join(out_dir, f'cam_XYZA_{trial_id}.h5'), \
                [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'), \
                 (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'), \
                 (cam_XYZA_pts.astype(np.float32), 'pc', 'float32'), \
                 ])

        object_link_ids = env.movable_link_ids
        gt_movable_link_mask = cam.get_movable_link_mask([env.movable_link_ids])

        target_part_id = object_link_ids[gt_movable_link_mask[part_x, part_y] - 1]

        print("target_part_id: ", target_part_id)
        env.set_target_object_part_actor_id(target_part_id)
        target_part_joint_idx = env.target_object_part_joint_id

        for i in env.object.get_joints():
            if i.get_child_link().get_id() == target_part_id:
                mass = i.get_child_link().get_mass()
                print("mass:", mass)
                out_info['mass'] = mass

        gt_movable_link_mask = cam.get_movable_link_mask([target_part_id])

        # Image.fromarray((gt_movable_link_mask > 0).astype(np.uint8) * 255).save(
        #     os.path.join(out_dir, f'interaction_mask{trial_id}.png'))
        xs, ys = np.where(gt_movable_link_mask > 0)


        # env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y] - 1])
        out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
        out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
        # damp = np.random.rand()*random_range
        damp = random_range
        for j in env.object.get_joints():
            if j.get_child_link().get_id() == env.target_object_part_actor_id:
                j.set_friction(friction)
        force = 0
        # succ_images = []
        ct_error = 0
        for i in range(3000):
            # print("inner cycle")
            print("shape id ",shape_id)
            not_still_cnt = 0
            while True:
                not_still_cnt += 1
                for j in range(cur_qpos.shape[0]):
                    if j == target_part_joint_idx:
                        if np.random.rand() > 0.5:
                            cur_qpos[j] = 0.1 + np.random.rand()*0.7
                        else :
                            cur_qpos[j] = env.joint_angles_lower[j]
                env.object.set_qpos(cur_qpos.astype(np.float32))
                still_timesteps = 0
                wait_timesteps = 0
                while still_timesteps < 150 and wait_timesteps < 200:
                    env.step()
                    env.render()
                    cur_new_qpos = env.get_object_qpos()
                    invalid_contact = False
                    for c in env.scene.get_contacts():
                        for p in c.points:
                            if abs(p.impulse @ p.impulse) > 1e-4:
                                invalid_contact = True
                                break
                        if invalid_contact:
                            break
                    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
                        still_timesteps += 1
                    else:
                        still_timesteps = 0
                    cur_qpos = cur_new_qpos
                    wait_timesteps += 1
                if still_timesteps < 150 and not_still_cnt < 20:
                    # env.scene.remove_articulation(env.object)
                    print("not still")
                    continue
                else :
                    break
            if not_still_cnt > 19:
                print('success rate : %f\n' % (succ_cnt / (i + 1)))
                env.scene.remove_articulation(env.object)
                break
            out_info['cur_qpos'] = cur_qpos.tolist()
            env.render()
            env.step()
            rgb, depth = cam.get_observation()
            # Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

            cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
            cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
            save_h5(os.path.join(out_dir, f'cam_XYZA_{trial_id}_{succ_cnt + 1}.h5'), \
                    [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'), \
                     (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'), \
                     (cam_XYZA_pts.astype(np.float32), 'pc', 'float32'), \
                     ])

            gt_movable_link_mask = cam.get_movable_link_mask([target_part_id])
            xs, ys = np.where(gt_movable_link_mask > 0)
            if (len(xs) == 0):
                print("cant see target part")
                continue

            ############################ random select ctpt ############################################
            xs, ys = np.where(gt_movable_link_mask > 0)
            idx = np.random.randint(len(xs))
            x, y = xs[idx], ys[idx]
            position_cam = cam_XYZA[x, y, :3]
            position_cam_xyz1 = np.ones((4), dtype=np.float32)
            position_cam_xyz1[:3] = position_cam
            position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
            position_world = position_world_xyz1[:3]
            ###########################################################################################

            ################################ random actor ##################################
            gt_nor = cam.get_normal_map()
            direction_cam = gt_nor[x, y, :3]
            direction_cam /= np.linalg.norm(direction_cam)
            out_info['direction_camera'] = direction_cam.tolist()
            direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
            out_info['direction_world'] = direction_world.tolist()

            action_direction_cam = np.random.randn(3).astype(np.float32)
            action_direction_cam /= np.linalg.norm(action_direction_cam)
            while action_direction_cam @ direction_cam > -0.1:
                action_direction_cam = np.random.randn(3).astype(np.float32)
                action_direction_cam /= np.linalg.norm(action_direction_cam)

            out_info['gripper_direction_camera'] = action_direction_cam.tolist()
            action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
            out_info['gripper_direction_world'] = action_direction_world.tolist()
            up = np.array(action_direction_world, dtype=np.float32)
            forward = np.random.randn(3).astype(np.float32)
            while abs(up @ forward) > 0.99:
                forward = np.random.randn(3).astype(np.float32)

            ################################################################################
            left = np.cross(up, forward)
            left /= np.linalg.norm(left)
            forward = np.cross(left, up)
            forward /= np.linalg.norm(forward)
            action_direction_world = up
            out_info['gripper_direction_world'] = action_direction_world.tolist()
            out_info['gripper_forward_direction_world'] = forward.tolist()


            robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
            out_info['mu1'] = mu1
            out_info['mu2'] = mu2
            out_info['mu3'] = mu3
            out_info['mu4'] = mu4
            out_info['damp'] = damp
            out_info['friction'] = friction
            out_info['density'] = density

            out_info['pixel_locs'] = np.array([x, y], dtype=np.int32).tolist()
            out_info['position_cam'] = position_cam.tolist()
            out_info['position_world'] = position_world.tolist()

            joint_origins = env.get_target_part_origins_new(target_part_id=target_part_id)
            state_joint_origins = joint_origins
            state_joint_origins[-1] = position_world[-1]
            state_ctpt_dis_to_joint = np.linalg.norm(state_joint_origins - position_world)
            state_joint_origins.append(state_ctpt_dis_to_joint)
            out_info['joint_info'] = state_joint_origins


            ################################# random actor ##################################
            gt_nor = cam.get_normal_map()
            direction_cam = gt_nor[x, y, :3]
            direction_cam /= np.linalg.norm(direction_cam)
            out_info['direction_camera'] = direction_cam.tolist()
            direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
            out_info['direction_world'] = direction_world.tolist()

            action_direction_cam = np.random.randn(3).astype(np.float32)
            action_direction_cam /= np.linalg.norm(action_direction_cam)
            while action_direction_cam @ direction_cam > -0.7:
                action_direction_cam = np.random.randn(3).astype(np.float32)
                action_direction_cam /= np.linalg.norm(action_direction_cam)

            out_info['gripper_direction_camera'] = action_direction_cam.tolist()
            action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
            out_info['gripper_direction_world'] = action_direction_world.tolist()
            up = np.array(action_direction_world, dtype=np.float32)
            forward = np.random.randn(3).astype(np.float32)
            while abs(up @ forward) > 0.99:
                forward = np.random.randn(3).astype(np.float32)

            left = np.cross(up, forward)
            left /= np.linalg.norm(left)
            forward = np.cross(left, up)
            forward /= np.linalg.norm(forward)
            out_info['gripper_forward_direction_world'] = forward.tolist()
            forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
            out_info['gripper_forward_direction_camera'] = forward_cam.tolist()
            #################################################################################
            rotmat = np.eye(4).astype(np.float32)
            rotmat[:3, 0] = forward
            rotmat[:3, 1] = left
            rotmat[:3, 2] = up

            final_dist = args.final_dist
            out_info['dist'] = final_dist

            final_rotmat = np.array(rotmat, dtype=np.float32)
            final_rotmat[:3, 3] = position_world - action_direction_world * final_dist - action_direction_world * 0.1
            final_pose = Pose().from_transformation_matrix(final_rotmat)
            out_info['target_rotmat_world'] = final_rotmat.tolist()

            start_rotmat = np.array(rotmat, dtype=np.float32)
            start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
            start_pose = Pose().from_transformation_matrix(start_rotmat)
            out_info['start_rotmat_world'] = start_rotmat.tolist()

            end_rotmat = np.array(rotmat, dtype=np.float32)
            end_rotmat[:3, 3] = position_world - action_direction_world * 0.1
            out_info['end_rotmat_world'] = end_rotmat.tolist()

            ### viz the EE gripper position
            # setup robot
            # robot.robot.set_root_pose(final_pose)
            # env.render()
            # move back
            robot.robot.set_root_pose(start_pose)
            env.render()

            # activate contact checking
            env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in primact_type)

            ### main steps
            out_info['start_target_part_qpos'] = float(env.get_object_qpos()[target_part_joint_idx])
            target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
            position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
            grasp_fail = True
            success = True
            try:
                init_success = True
                success_grasp = False
                print("try to grasp")
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
                    grasp_fail = True
                else :
                    try:
                        robot.move_to_target_pose(final_rotmat, 2000)
                        robot.wait_n_steps(2000)

                    except Exception:
                        print("fail")
                        ct_error = ct_error + 1
                        success = False
                # env.end_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in primact_type)
                # succ_images.extend(imgs)
                # imageio.mimsave(os.path.join(out_dir, 'pic_%s_%s.gif' % (trial_id, i)), succ_images)


            except Exception:
                success = False

            target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
            position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
            out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
            out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()
            env.scene.remove_articulation(robot.robot)
            succ_cnt = succ_cnt + 1
            if success:
                out_info['final_target_part_qpos'] = float(env.get_object_qpos()[target_part_joint_idx])
                out_info['gt_labels'] = 0

                tot_motion = out_info['joint_angles_upper'][target_part_joint_idx] - out_info['joint_angles_lower'][target_part_joint_idx] + 1e-8
                gt_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])

                mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
                          np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32)
                mov_dir /= np.linalg.norm(mov_dir)
                intended_dir = -np.array(out_info['gripper_direction_world'], dtype=np.float32)
                intend_motion = intended_dir @ mov_dir

                if (gt_motion > args.true_thres or gt_motion/tot_motion > 0.5) and intend_motion > 0.5:

                    out_info['gt_labels'] = 1
                    with open(os.path.join(out_dir, 'result_%d_%d.json' % (trial_id, succ_cnt)), 'w') as fout:
                        json.dump(out_info, fout)
                elif (succ_cnt > 7 or i > 1000):
                    out_info['gt_labels'] = 0
                    with open(os.path.join(out_dir, 'result_%d_%d.json' % (trial_id, succ_cnt)), 'w') as fout:
                        json.dump(out_info, fout)
                else :
                    succ_cnt -= 1
            elif succ_cnt > 15 or i > 1500:
                out_info['final_target_part_qpos'] = out_info['start_target_part_qpos']
                out_info['gt_labels'] = 0
                with open(os.path.join(out_dir, 'result_%d_%d.json' % (trial_id, succ_cnt)), 'w') as fout:
                    json.dump(out_info, fout)
            else:
                succ_cnt = succ_cnt - 1


            if succ_cnt > 32 or i + 1 == 3000 or (i>2000 and succ_cnt < 4):
                print('success rate : %f\n' % (succ_cnt / (i + 1)))
                env.scene.remove_articulation(env.object)
                break

        # env.scene.remove_articulation(env.object)
        # end of trial
        if (succ_cnt < 32):
            break
        trial_id = trial_id + 1
    # close the file
    flog.close()
    env.close()


for idx_process in range(args.num_processes):
    run_collect(idx_process, args.trial_id, args.shape_id, args.primact_type, args.out_dir)