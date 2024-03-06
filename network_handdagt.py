
import torch
import torch.nn as nn
import math
import numpy as np
from pointutil import Conv1d,  Conv2d, PointNetSetAbstraction, BiasConv1d, square_distance, index_points_group
import torch.nn.functional as F
from convNeXT.resnetUnet import convNeXTUnetBig

def smooth_l1_loss(input, target, sigma=10., reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer
criterion = smooth_l1_loss

model_list = {
          'tiny': ([3, 3, 9, 3], [96, 192, 384, 768]),
          'small': ([3, 3, 27, 3], [96, 192, 384, 768]),
          'base': ([3, 3, 27, 3], [128, 256, 512, 1024]),
          'large': ([3, 3, 27, 3], [192, 384, 768, 1536])
          }
weight_url_1k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224.pth"
}

weight_url_22k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth"
}

class GAT_GCN(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, graph_width, mlp, mlp2=None, bn = False, use_leaky = True, return_inter=False, radius=None, relu=False, bypass_gcn=False, bias=True, graph_bias=True):
        super(GAT_GCN,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_q_convs = nn.ModuleList()
        self.mlp_g_convs = nn.ModuleList()
        self.mlp_v_convs = nn.ModuleList()
        self.mlp_k_convs = nn.ModuleList()
        self.mlp_q_bns = nn.ModuleList()
        self.mlp_g_bns = nn.ModuleList()
        self.mlp_v_bns = nn.ModuleList()
        self.mlp_k_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu
        self.bypass_gcn = bypass_gcn

        last_channel = in_channel + 3

        self.graph_a = nn.Parameter(torch.randn(1, latent_channel, graph_width, graph_width).cuda(),requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(latent_channel, latent_channel, 1, bias=graph_bias),
                                     nn.BatchNorm1d(latent_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True))
        self.fuse_q = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_v1 = nn.Conv2d(last_channel, mlp[0], 1, bias=False)
        self.fuse_v2 = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_k = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        # self.fuse_g = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for i, out_channel in enumerate(mlp):
            self.mlp_q_convs.append(nn.Conv2d(last_channel, out_channel if i < len(mlp)-1 else out_channel * 2, 1, bias=bias))
            # self.mlp_g_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=bias))
            self.mlp_v_convs.append(nn.Conv2d(last_channel if i > 0 else mlp[0], out_channel, 1, bias=bias))
            self.mlp_k_convs.append(nn.Conv2d(last_channel, out_channel if i < len(mlp)-1 else out_channel * 2, 1, bias=bias))
            if bn:
                self.mlp_q_bns.append(nn.BatchNorm2d(out_channel if i < len(mlp)-1 else out_channel * 2))
                # self.mlp_g_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_v_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_k_bns.append(nn.BatchNorm2d(out_channel if i < len(mlp)-1 else out_channel * 2))
            last_channel = out_channel
        
        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.atten_bn = nn.BatchNorm2d(out_channel)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_v
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
  
        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim = -1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx) # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1) # [B, D2+3, nsample, N1]
        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2),self.graph_a).squeeze(-2))

        # q
        q = new_points
        for i, conv in enumerate(self.mlp_q_convs):
            q = conv(q)
            if i == 0:
                grouped_points1 = self.fuse_q(point1_graph)
                q = q + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                q = self.mlp_q_bns[i](q)
            if i == len(self.mlp_q_convs) - 1:
                q = q
            else:
                q = self.relu(q)

        k = new_points
        for i, conv in enumerate(self.mlp_k_convs):
            k = conv(k)
            if i == 0:
                grouped_points1 = self.fuse_k(point1_graph)
                k = k + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                k = self.mlp_k_bns[i](k)
            if i == len(self.mlp_k_convs) - 1:
                k = k
            else:
                k = self.relu(k)
            if i == len(self.mlp_k_convs) - 2:
                k = torch.max(k, -2)[0].unsqueeze(-2)
        
        a = self.sigmoid(k*q)

        g1, g2 = torch.chunk(a, 2, 1)

        v = new_points
        v = self.fuse_v1(v)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = g2 * point1_graph_expand
        point1_expand = self.fuse_v2(point1_expand)

        v = self.relu(v * g1 + point1_expand) + points1.unsqueeze(2).repeat(1, 1, self.nsample, 1)
        v_res = v.mean(2)

        for i, conv in enumerate(self.mlp_v_convs):
            v = conv(v)
            if i == 0:
                v = v * g1 + point1_expand
            if self.bn:
                v = self.mlp_v_bns[i](v)
            if i == len(self.mlp_v_convs) - 1:
                v = self.relu(v)
            else:
                v = self.relu(v)
            if i == len(self.mlp_v_convs) - 2:
                v = torch.max(v, -2)[0].unsqueeze(-2)

        v = v.squeeze(-2)

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)        

        return v + v_res


class HandModel(nn.Module):
    def __init__(self, joints=21, stacks=10):
        super(HandModel, self).__init__()
        
        self.backbone = convNeXTUnetBig('small', pretrain='1k', deconv_dim=128)

        self.encoder_1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=64, in_channel=3, mlp=[32,32,128])
        
        self.encoder_2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=64, in_channel=128, mlp=[64,64,256])

        self.encoder_3 = nn.Sequential(Conv1d(in_channels=256+3, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=512, bn=True, bias=False),
                                       nn.MaxPool1d(128,stride=1))

        self.fold1 = nn.Sequential(BiasConv1d(bias_length=joints, in_channels=512+768, out_channels=512, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True))
        self.regress_1 = nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)

        self.trans = nn.ModuleList([GAT_GCN(nsample=64, in_channel=128+128, latent_channel=512, graph_width=joints, mlp=[512, 512, 512], mlp2=None) for _ in range(stacks)])
        self.regress = nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)

        self.stacks = stacks
        self.joints = joints

    def log_snr(self, t):
        return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))
    def log_snr_to_alpha_sigma(self, log_snr):
        return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

    def encode(self, pc, feat, img, loader, center, M, cube, cam_para):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1

        pc1, feat1 = self.encoder_1(pc, feat)# B, 3, 512; B, 64, 512
        
        pc2, feat2 = self.encoder_2(pc1, feat1)# B, 3, 256; B, 128, 256
        
        code = self.encoder_3(torch.cat((pc2, feat2),1))# B, 3, 128; B, 256, 128
        
        pc_img_feat, c4 = self.backbone(img)   # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
        img_code = torch.max(c4.view(c4.size(0),c4.size(1),-1),-1,keepdims=True)[0]
        B, C, H, W = pc_img_feat.size()
        img_down = F.interpolate(img, [H, W])
        B, _, N = pc1.size()

        pcl_closeness, pcl_index, img_xyz = loader.img2pcl_index(pc1.transpose(1,2).contiguous(), img_down, center, M, cube, cam_para, select_num=4)

        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)   # B*128*(K*1024)
        pcl_feat = torch.gather(pc_img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat = torch.sum(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1)
        feat1 = torch.cat((feat1,pcl_feat),1)

        code = code.expand(code.size(0),code.size(1), self.joints)
        img_code = img_code.expand(img_code.size(0),img_code.size(1), self.joints)

        latents = self.fold1(torch.cat((code, img_code),1))
        joints = self.regress_1(latents)

        return latents, joints, pc1, feat1, img_xyz, pc_img_feat

    def forward(self, pc, feat, img, loader, center, M, cube, cam_para):
        embed, joint, pc1, feat1, _, _ = self.encode(pc, feat, img, loader, center, M, cube, cam_para)

        # joint = torch.randn_like(joint) * 0.1 + joint  # (B, d, J) original training
        # joint = torch.randn_like(joint) * 3 + joint  # (B, d, J)

        for i in range(self.stacks):
            embed = self.trans[i](joint, pc1, embed, feat1)
            joint = self.regress(embed)
        return joint

    def get_loss(self, pc, feat, img, loader, center, M, cube, cam_para, gt):
        embed, joint, pc1, feat1, _, _ = self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        loss = smooth_l1_loss(joint, gt)

        times = torch.zeros(
            (joint.size(0),), device=joint.device).float().uniform_(0.5, 1)
        log_snr = self.log_snr(times)
        alpha, sigma = self.log_snr_to_alpha_sigma(times)
        # c0 = alpha.view(-1, 1, 1)   # (B, 1, 1)
        c1 = torch.sqrt(torch.sigmoid(-log_snr)).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(joint)  # (B, d, J)
        joint = joint + c1 * e_rand

        for i in range(self.stacks):
            embed = self.trans[i](joint, pc1, embed, feat1)
            joint = self.regress(embed)
            loss += smooth_l1_loss(joint, gt)

        return loss

