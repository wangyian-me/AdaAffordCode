import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ipdb

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

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


class Affordance(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(Affordance, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})
        self.mlp3 = nn.Linear(hidden_dim, feat_dim)
        self.mlp1 = nn.Linear(feat_dim + feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')


    def forward(self, pcs, hidden_info):
        # pcs[:, 0] = contact_point
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]
        hidden_info = self.mlp3(hidden_info)
        net = torch.cat([net, hidden_info], dim=-1)
        net = self.mlp1(net)
        net = F.leaky_relu(net)
        net = self.mlp2(net).squeeze(-1)
        return net

    def inference_whole_pc(self, input_pcs, hidden_info):
        batch_size = input_pcs.shape[0]
        pt_size = input_pcs.shape[1]
        hidden_info = hidden_info.view(batch_size, -1)
        hidden_info = self.mlp3(hidden_info)
        hidden_info = hidden_info.unsqueeze(1).repeat(1, pt_size, 1).view(batch_size * pt_size, -1)
        input_pcs = input_pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(input_pcs)
        net = whole_feats.permute(0, 2, 1).reshape(batch_size * pt_size, -1)
        x = torch.cat([net, hidden_info], dim=-1)
        x = self.mlp1(x)
        x = F.leaky_relu(x)
        pred_result_logits = self.mlp2(x).squeeze(-1)
        # pred_result_logits = torch.sigmoid(pred_result_logits)
        return pred_result_logits

    def get_loss(self, pcs, hidden_info, gt_score):
        batch_size = pcs.shape[0]
        score = self.forward(pcs, hidden_info)
        loss = self.L1Loss(score.view(-1), gt_score).mean()
        return loss

    def get_loss_new(self, pc_feats, hidden_info, gt_score):
        hidden_info = self.mlp3(hidden_info)
        net = torch.cat([pc_feats, hidden_info], dim=-1)
        net = self.mlp1(net)
        net = F.leaky_relu(net)
        net = self.mlp2(net).squeeze(-1)
        loss = self.L1Loss(net.view(-1), gt_score).mean()
        return loss
