import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader

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


class AIP(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(AIP, self).__init__()

        self.mlp3 = nn.Linear(6, feat_dim)
        self.mlp4 = nn.Linear(1, feat_dim)
        self.mlp1 = nn.Linear(feat_dim + hidden_dim + feat_dim + feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.L1_criterion = nn.L1Loss(reduction="none")

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, input_pcs, hidden_info, dir, f_dir, critic_score):
        batch_size = hidden_info.shape[0]
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
        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = F.leaky_relu(x)
        y = torch.cat([critic_score], dim=-1)
        y = self.mlp4(y)
        y = F.leaky_relu(y)
        batch_size = x.shape[0]
        y = y.view(batch_size, -1)
        # print("x_shape:", x.shape)
        # print("y_shape:", y.shape)
        # print("hidden_shape:", hidden_info.shape)
        # print("pc_shape:", pc_feat.shape)
        z = torch.cat([x, y, hidden_info, pc_feat], dim=-1)
        z = self.mlp1(z)
        z = F.leaky_relu(z)
        z = self.mlp2(z).squeeze(-1)
        return z
    # cross entropy loss
    # def get_ce_loss(self, pred_logits, gt_labels):
    def get_loss(self, input_pcs, gt_scores, hidden_info, dir, f_dir, c_score):
        pred_scores = self.forward(input_pcs, hidden_info, dir, f_dir, c_score)
        return self.L1_criterion(pred_scores, gt_scores).mean()

    def inference_score(self, input_pcs, hidden_info, dir, f_dir, c_score):
        batch_size = input_pcs.shape[0]
        pt_size = input_pcs.shape[1]
        hidden_info = hidden_info.view(batch_size, -1)
        hidden_info = hidden_info.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        dir = dir.view(batch_size * pt_size, -1)
        f_dir = f_dir.view(batch_size * pt_size, -1)
        # ctpt = input_pcs.view(batch_size * pt_size, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        c_score = c_score.view(batch_size * pt_size, -1)
        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = F.leaky_relu(x)
        y = torch.cat([c_score], dim=-1)
        y = self.mlp4(y)
        y = F.leaky_relu(y)
        # ipdb.set_trace()
        z = torch.cat([x, y, hidden_info, net], dim=-1)
        z = self.mlp1(z)
        z = F.leaky_relu(z)
        z = self.mlp2(z).squeeze(-1)
        return z
    def inference_nscore(self, input_pcs, hidden_info, dir, f_dir, c_score, n):
        batch_size = input_pcs.shape[0]
        pt_size = input_pcs.shape[1]
        hidden_info = hidden_info.view(batch_size, -1)
        hidden_info = hidden_info.unsqueeze(1).repeat(1, pt_size * n, 1).view(batch_size * pt_size * n, -1)
        dir = dir.view(batch_size * pt_size * n, -1)
        f_dir = f_dir.view(batch_size * pt_size * n, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        net = net.unsqueeze(dim=1).repeat(1, n, 1).reshape(batch_size * pt_size * n, -1)

        c_score = c_score.view(batch_size * pt_size * n, -1)
        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = F.leaky_relu(x)
        y = torch.cat([c_score], dim=-1)
        y = self.mlp4(y)
        y = F.leaky_relu(y)
        # ipdb.set_trace()
        z = torch.cat([x, y, hidden_info, net], dim=-1)
        z = self.mlp1(z)
        z = F.leaky_relu(z)
        z = self.mlp2(z).squeeze(-1)
        return z
    def inference_naction(self, dir, input_pcs, f_dir, hidden_info, action_scores, rvs_cnt):
        batch_size = input_pcs.shape[0]
        hidden_info = hidden_info.view(batch_size, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats[:, :, 0]
        net = net.repeat(1, rvs_cnt, 1).view(batch_size * rvs_cnt, -1)
        hidden_info = hidden_info.repeat(1, rvs_cnt).view(batch_size * rvs_cnt, -1)
        action_scores = action_scores.view(batch_size * rvs_cnt, -1)
        dir = dir.view(batch_size * rvs_cnt, -1)
        f_dir = f_dir.view(batch_size * rvs_cnt, -1)
        x = torch.cat([dir, f_dir], dim=-1)
        x = self.mlp3(x)
        x = F.leaky_relu(x)
        y = torch.cat([action_scores], dim=-1)
        y = self.mlp4(y)
        y = F.leaky_relu(y)
        # ipdb.set_trace()
        z = torch.cat([x, y, hidden_info, net], dim=-1)
        z = self.mlp1(z)
        z = F.leaky_relu(z)
        z = self.mlp2(z).squeeze(-1)
        return z.reshape(batch_size, rvs_cnt, 1)
