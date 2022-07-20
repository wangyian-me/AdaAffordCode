import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import random
# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
import numpy as np

import ipdb

class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )
        # self.SA_modules.append(
        #     PointnetSAModule(
        #         mlp=[512, 512],
        #     )
        # )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        # print(pointcloud.shape)
        # print(xyz.shape)
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            # print(li_features.shape)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class Encoder(nn.Module):
    def __init__(self, hidden_dim=2, input_dim=16, pnpp_feat_dim=128, hidden_feat_dim=128):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_feat_dim + pnpp_feat_dim, hidden_feat_dim),
            nn.ReLU()
        )
        self.hidden_info_encoder = nn.Linear(hidden_feat_dim, hidden_dim)

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': pnpp_feat_dim})
        self.AttentionNet = nn.Sequential(
            nn.Linear(hidden_feat_dim, hidden_feat_dim),
            nn.ReLU(),
            nn.Linear(hidden_feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, dir, dis, push_dis, ctpt, joint_info, pcs, start_pos, end_pos, f_dir):
        batch_size = dir.shape[0]
        dir = dir.view(batch_size, -1)
        f_dir = f_dir.view(batch_size, -1)
        dis = dis.view(batch_size, -1)
        ctpt = ctpt.view(batch_size, -1)
        start_pos = start_pos.view(batch_size, -1)
        end_pos = end_pos.view(batch_size, -1)
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        pc_feat = whole_feats[:, :, 0]

        joint_info = joint_info.view(batch_size, -1)
        x = torch.cat([dir, dis, push_dis, ctpt, joint_info, start_pos, end_pos, f_dir], dim=-1)
        hidden_feat = self.mlp1(x)
        hidden_feat = torch.cat([hidden_feat, pc_feat], dim=1)
        hidden_feat = self.mlp2(hidden_feat)
        hidden_info = self.hidden_info_encoder(hidden_feat)
        hidden_info_attention = self.AttentionNet(hidden_feat)

        hidden_info = hidden_info * hidden_info_attention / (0.00001 + hidden_info_attention.sum())
        mean_hidden_info = hidden_info.sum(dim=0).view(1, self.hidden_dim)

        return mean_hidden_info
    def get_attention_score(self, dir, dis, push_dis, ctpt, joint_info, pcs, start_pos, end_pos, f_dir):
        batch_size = dir.shape[0]
        dir = dir.view(batch_size, -1)
        f_dir = f_dir.view(batch_size, -1)
        dis = dis.view(batch_size, -1)
        ctpt = ctpt.view(batch_size, -1)
        start_pos = start_pos.view(batch_size, -1)
        end_pos = end_pos.view(batch_size, -1)
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        pc_feat = whole_feats[:, :, 0]

        joint_info = joint_info.view(batch_size, -1)
        x = torch.cat([dir, dis, push_dis, ctpt, joint_info, start_pos, end_pos, f_dir], dim=-1)
        hidden_feat = self.mlp1(x)
        hidden_feat = torch.cat([hidden_feat, pc_feat], dim=1)
        hidden_feat = self.mlp2(hidden_feat)
        hidden_info_attention = self.AttentionNet(hidden_feat)
        return hidden_info_attention


class Critic(nn.Module):
    def __init__(self, feat_dim, hidden_dim = 2):
        super(Critic, self).__init__()
        self.mlp3 = nn.Linear(3 + 3 + 1 + 3 + 3 + 1 + hidden_dim, feat_dim)
        self.mlp1 = nn.Linear(feat_dim + feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, st_pose):
        batch_size = dir.shape[0]
        # hidden_info = hidden_info.repeat(batch_size, 1)
        hidden_info = hidden_info.view(batch_size, -1)
        pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        pc_feat = whole_feats[:, :, 0]
        # feats for the interacting points
        # pixel_feats = whole_feats[:, :, 0]  # B x F
        # one_net = torch.ones(query_feats.shape[:-1]).unsqueeze(-1).to(query_feats.device).float()
        # net = torch.cat([pixel_feats, query_feats, one_net,hidden_info,ctpt], dim=-1)
        # net = F.leaky_relu(self.mlp1(net))
        # net = self.mlp2(net).squeeze(-1)
        # return net
        x = torch.cat([dir, ctpt, joint_info, f_dir, hidden_info, st_pose], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([pc_feat, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        x = self.mlp2(x).squeeze(-1)
        return x
    # cross entropy loss
    # def get_ce_loss(self, pred_logits, gt_labels):
    def get_ce_loss(self, dir, input_pcs, ctpt, joint_info, f_dir, gt_labels, hidden_info, st_pose):
        pred_logits = self.forward(dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, st_pose)
        loss = self.BCELoss(pred_logits, gt_labels)
        return loss.mean()

    def inference_critic_score(self, dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, st_pose):
        batch_size = input_pcs.shape[0]
        pt_size = input_pcs.shape[1]
        hidden_info = hidden_info.view(batch_size, -1)
        hidden_info = hidden_info.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        dir = dir.view(batch_size, -1)
        dir = dir.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        st_pose = st_pose.view(batch_size, -1)
        st_pose = st_pose.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        f_dir = f_dir.view(batch_size, -1)
        f_dir = f_dir.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        ctpt = input_pcs.view(batch_size * pt_size, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        x = torch.cat([dir, ctpt, joint_info, f_dir, hidden_info, st_pose], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([net, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        pred_result_logits = self.mlp2(x).squeeze(-1)
        pred_result_logits = torch.sigmoid(pred_result_logits)
        return pred_result_logits

    def inference_critic_score_diff_action(self, dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, st_pose):
        batch_size = input_pcs.shape[0]
        pt_size = input_pcs.shape[1]
        hidden_info = hidden_info.view(batch_size, -1)
        hidden_info = hidden_info.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        dir = dir.view(batch_size * pt_size, -1)
        st_pose = st_pose.view(batch_size, -1)
        st_pose = st_pose.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        f_dir = f_dir.view(batch_size * pt_size, -1)
        ctpt = input_pcs.view(batch_size * pt_size, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        x = torch.cat([dir, ctpt, joint_info, f_dir, hidden_info, st_pose], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([net, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        pred_result_logits = self.mlp2(x).squeeze(-1)
        pred_result_logits = torch.sigmoid(pred_result_logits)
        return pred_result_logits

    def inference_critic_score_diff_naction(self, dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, st_pose, n):
        batch_size = input_pcs.shape[0]
        pt_size = input_pcs.shape[1]
        hidden_info = hidden_info.view(batch_size, -1)
        hidden_info = hidden_info.unsqueeze(1).repeat(1, pt_size * n, 1).view(batch_size * pt_size * n, -1)
        dir = dir.view(batch_size * pt_size*n, -1)
        st_pose = st_pose.view(batch_size, -1)
        st_pose = st_pose.unsqueeze(1).repeat(1, pt_size*n, 1).view(batch_size * pt_size*n, -1)
        f_dir = f_dir.view(batch_size * pt_size*n, -1)
        ctpt = input_pcs.view(batch_size * pt_size, -1)
        ctpt = ctpt.unsqueeze(dim=1).repeat(1, n, 1).reshape(batch_size * pt_size * n, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        net = net.unsqueeze(dim=1).repeat(1, n, 1).reshape(batch_size * pt_size * n, -1)
        x = torch.cat([dir, ctpt, joint_info, f_dir, hidden_info, st_pose], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([net, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        pred_result_logits = self.mlp2(x).squeeze(-1)
        # pred_result_logits = pred_result_logits.view(batch_size*pt_size, n, 1).topk(k=3, dim=1)[0].mean(dim=1).view(-1)
        pred_result_logits = torch.sigmoid(pred_result_logits)
        return pred_result_logits


    def inference_naction(self, dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, st_pose, rvs_cnt):
        batch_size = input_pcs.shape[0]
        hidden_info = hidden_info.view(batch_size, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats[:, :, 0]
        net = net.repeat(1, rvs_cnt, 1).view(batch_size * rvs_cnt, -1)
        ctpt = ctpt.repeat(1, rvs_cnt, 1).view(batch_size * rvs_cnt, -1)
        hidden_info = hidden_info.repeat(1, rvs_cnt).view(batch_size * rvs_cnt, -1)
        joint_info = joint_info.repeat(1, rvs_cnt, 1).view(batch_size * rvs_cnt, -1)
        st_pose = st_pose.view(batch_size, -1)
        st_pose = st_pose.repeat(1, rvs_cnt, 1).view(batch_size * rvs_cnt, -1)
        dir = dir.view(batch_size * rvs_cnt, -1)
        f_dir = f_dir.view(batch_size * rvs_cnt, -1)
        x = torch.cat([dir, ctpt, joint_info, f_dir, hidden_info, st_pose], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([net, x], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        pred_result_logits = self.mlp2(x).squeeze(-1)
        # pred_result_logits = torch.sigmoid(pred_result_logits)
        return pred_result_logits.reshape(batch_size, rvs_cnt, 1)


class network(nn.Module):
    def __init__(self, input_dim=16, pnpp_feat_dim=128, hidden_feat_dim=128, feat_dim=128, hidden_dim = 2):
        super(network, self).__init__()

        self.hidden_encoder = Encoder(hidden_dim=hidden_dim, input_dim=input_dim, pnpp_feat_dim=pnpp_feat_dim, hidden_feat_dim=hidden_feat_dim)
        self.critic = Critic(feat_dim=feat_dim, hidden_dim=hidden_dim)

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, dir, input_pcs, ctpt, joint_info, f_dir, hidden_info):
        return dir
    # cross entropy loss
    # def get_ce_loss(self, pred_logits, gt_labels):
    def get_loss(self, dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir, gt_labels):
        batch_size = input_pcs.shape[0]
        mean_hidden_info = self.hidden_encoder(dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir)
        hidden_info = mean_hidden_info.repeat(batch_size, 1)
        loss = self.critic.get_ce_loss(dir, input_pcs, ctpt, joint_info, f_dir, gt_labels, hidden_info, start_pos)
        return loss
    def get_loss_dropout(self, dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir, gt_labels, dropout, dropout_cnt=16):
        batch_size = input_pcs.shape[0]
        idxl = np.array(random.sample([0,2,4,6,8,10,12,14], dropout_cnt))
        idxr = np.array(random.sample([1,3,5,7,9,11,13,15], dropout_cnt))
        # idxr = np.array(random.sample([21,22,23,24,25,26,27,28,29,30], 1))
        if dropout == 'left':
            mean_hidden_info = self.hidden_encoder(dir[idxl,:], dis[idxl], push_dis[idxl], ctpt[idxl,:], joint_info[idxl,:], input_pcs[idxl,:], start_pos[idxl], end_pos[idxl], f_dir[idxl])
        elif dropout == 'right':
            mean_hidden_info = self.hidden_encoder(dir[idxr,:], dis[idxr], push_dis[idxr], ctpt[idxr,:], joint_info[idxr,:], input_pcs[idxr,:], start_pos[idxr], end_pos[idxr], f_dir[idxr])
        elif dropout == 'random':
            idx = np.arange(batch_size)
            np.random.shuffle(idx)
            idx = idx[:dropout_cnt]
            mean_hidden_info = self.hidden_encoder(dir[idx, :], dis[idx], push_dis[idx], ctpt[idx, :], joint_info[idx, :],
                                             input_pcs[idx, :], start_pos[idx], end_pos[idx], f_dir[idx])
        else :
            mean_hidden_info = self.hidden_encoder(dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir)
        hidden_info = mean_hidden_info.repeat(batch_size, 1)
        loss = self.critic.get_ce_loss(dir, input_pcs, ctpt, joint_info, f_dir, gt_labels, hidden_info, start_pos)
        return loss
    def inference(self, dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir):
        batch_size = input_pcs.shape[0]
        mean_hidden_info = self.hidden_encoder(dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir)
        hidden_info = mean_hidden_info.repeat(batch_size, 1)
        return self.critic(dir, input_pcs, ctpt, joint_info, f_dir, hidden_info, start_pos)
    def inference_part(self, dir, dis, push_dis, ctpt, joint_info, input_pcs, start_pos, end_pos, f_dir, idx1, idx2):
        batch_size = idx2.shape[0]
        mean_hidden_info = self.hidden_encoder(dir[idx1], dis[idx1], push_dis[idx1], ctpt[idx1], joint_info[idx1], input_pcs[idx1], start_pos[idx1], end_pos[idx1], f_dir[idx1])
        hidden_info = mean_hidden_info.repeat(batch_size, 1)
        return self.critic(dir[idx2], input_pcs[idx2], ctpt[idx2], joint_info[idx2], f_dir[idx2], hidden_info, start_pos[idx2])