import torch

from .vfe_template import VFETemplate
from .image_vfe_modules import ffn, f2v
import torch.nn as nn
import torch.nn.functional as F


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def convbn_3dNoBN(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.GroupNorm(2, out_planes))

class hourglassNoBN(nn.Module):
    def __init__(self, inplanes):
        super(hourglassNoBN, self).__init__()

        self.conv1 = nn.Sequential(convbn_3dNoBN(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3dNoBN(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3dNoBN(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3dNoBN(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.interpolate(pre, (presqu.shape[-3], presqu.shape[-2], presqu.shape[-1]), mode='trilinear', align_corners=True)
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(
                        F.interpolate(self.conv5(out), (presqu.shape[-3], presqu.shape[-2], presqu.shape[-1]), mode='trilinear', align_corners=True)
                          + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(
                        F.interpolate(self.conv5(out), (pre.shape[-3], pre.shape[-2], pre.shape[-1]), mode='trilinear', align_corners=True)
                          + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        out = F.interpolate(out, (x.shape[-3], x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=True)
        return out, pre, post

class ImageVFE(VFETemplate):
    def __init__(self, model_cfg, grid_size, point_cloud_range, depth_downsample_factor, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.grid_size = grid_size
        self.pc_range = point_cloud_range
        self.downsample_factor = depth_downsample_factor
        self.module_topology = [
            'ffn', 'f2v'
        ]
        # self.module_topology = [
        #     'ffn'
        # ]
        self.build_modules()


        self.voxel_size = model_cfg.VOXEL_SIZE
        self.init_dres = nn.Sequential(nn.Conv3d(16, 16, 3, 1, 1),
                                        nn.ReLU(inplace=True))
        self.dres0 = hourglassNoBN(16)
        self.dres1 = hourglassNoBN(16)

        self.occupancy3D_prob_0 = nn.Sequential(nn.Conv3d(16, 16, 3, 1, 1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(16, 1, 3, 1, 1))

        self.forward_ret_dict = {}
        self.bce = nn.BCELoss()
        self.bce_noReduce = nn.BCELoss(reduction='none')


    def build_modules(self):
        """
        Builds modules
        """
        for module_name in self.module_topology:
            module = getattr(self, 'build_%s' % module_name)()
            self.add_module(module_name, module)

    def build_ffn(self):
        """
        Builds frustum feature network
        Returns:
            ffn_module: nn.Module, Frustum feature network
        """
        ffn_module = ffn.__all__[self.model_cfg.FFN.NAME](
            model_cfg=self.model_cfg.FFN,
            downsample_factor=self.downsample_factor
        )
        self.disc_cfg = ffn_module.disc_cfg
        return ffn_module

    def build_f2v(self):
        """
        Builds frustum to voxel transformation
        Returns:
            f2v_module: nn.Module, Frustum to voxel transformation
        """
        f2v_module = f2v.__all__[self.model_cfg.F2V.NAME](
            model_cfg=self.model_cfg.F2V,
            grid_size=self.grid_size,
            pc_range=self.pc_range,
            disc_cfg=self.disc_cfg
        )
        return f2v_module

    def get_output_feature_dim(self):
        """
        Gets number of output channels
        Returns:
            out_feature_dim: int, Number of output channels
        """
        out_feature_dim = self.ffn.get_output_feature_dim()
        return out_feature_dim


    def forward(self, batch_dict, **kwargs):
        batch_dict = self.ffn(batch_dict)
        batch_dict = self.f2v(batch_dict)

        cost0 = batch_dict["voxel_features"]
        cost0 = self.init_dres(cost0)
        out1, pre1, post1 = self.dres0(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres1(out1, pre1, post1)
        out2 = out2 + cost0
        voxel_features = out2

        occupancy_prob_0 = self.occupancy3D_prob_0(voxel_features)
        occupancy_prob_0 = torch.clamp(torch.sigmoid(occupancy_prob_0), 1e-5, 1 - 1e-5)
        batch_dict["occupancy_prob_0"] = occupancy_prob_0
        batch_dict["voxel_features"] = voxel_features * batch_dict["occupancy_prob_0"]

        if self.training:
            self.forward_ret_dict["frustum_features_prob"] = batch_dict["frustum_features_prob"].squeeze(1)
            self.forward_ret_dict["occupancy_prob_0"] = batch_dict["occupancy_prob_0"].squeeze(1)
            self.forward_ret_dict["aug_lidar_coor_map"] = batch_dict["aug_lidar_coor_map"]

            '''frustum depth'''
            self.forward_ret_dict["depth_maps"] = batch_dict["depth_maps"]

        return batch_dict



    def occupancy_loss(self, depth_maps, frustum_features_prob, **kwargs):
        def bin_ray_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
            """
            Converts depth map into bin indices
            Args:
                depth_map: (H, W), Depth Map
                mode: string, Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                    UD: Uniform discretiziation
                    LID: Linear increasing discretiziation
                    SID: Spacing increasing discretiziation
                depth_min: float, Minimum depth value
                depth_max: float, Maximum depth value
                num_bins: int, Number of depth bins
                target: bool, Whether the depth bins indices will be used for a target tensor in loss comparison
            Returns:
                indices: (H, W), Depth bin indices
            """
            import math
            if mode == "UD":
                bin_size = (depth_max - depth_min) / num_bins
                indices = ((depth_map - depth_min) / bin_size)
            elif mode == "LID":
                bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
                indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
            elif mode == "SID":
                indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                          (math.log(1 + depth_max) - math.log(1 + depth_min))
            else:
                raise NotImplementedError

            if target:
                # Remove indicies outside of bounds
                mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
                indices[mask] = num_bins

                # Convert to integer
                indices = indices.type(torch.int64)

            B, H, W = indices.shape
            gt_one_hot = torch.zeros((B, num_bins + 1, H, W), device=indices.device)
            bin_ind = torch.arange(num_bins + 1,
                                   device=indices.device).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W)
            indices_repeat = indices.unsqueeze(1).repeat(1, num_bins + 1, 1, 1)
            gt_one_hot[bin_ind == indices_repeat] = 1.
            gt_one_hot[bin_ind > indices_repeat] = 2.
            return gt_one_hot

        disc_cfg = {
            "mode": "LID",
            "num_bins": 80,
            "depth_min": 2.0,
            "depth_max": 46.8
        }
        depth_target = bin_ray_depths(depth_maps, **disc_cfg, target=True)

        cls_loss = self.focal_loss(frustum_features_prob, depth_target[:, :-1, ...])

        tb_dict = {"occupancy_loss": cls_loss.item()}
        return cls_loss, tb_dict


    def occupancy_label_1213(self, points):

        VOXEL_SIZE = self.voxel_size
        POINT_CLOUD_RANGE = self.pc_range
        center = [(0-POINT_CLOUD_RANGE[2])/VOXEL_SIZE[2],
                  (0-POINT_CLOUD_RANGE[1])/VOXEL_SIZE[1],
                  (0-POINT_CLOUD_RANGE[0])/VOXEL_SIZE[0]]


        voxel_x_range = (POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0] + 1e-4) / VOXEL_SIZE[0]
        voxel_y_range = (POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1] + 1e-4) / VOXEL_SIZE[1]
        voxel_z_range = (POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2] + 1e-4) / VOXEL_SIZE[2]
        grid_3d = torch.zeros((int(voxel_x_range), int(voxel_y_range), int(voxel_z_range)), device=points.device)


        valid_ind = (points[:, 0] > POINT_CLOUD_RANGE[0] + 1e-4) & (points[:, 0] < POINT_CLOUD_RANGE[3] - (1e-4)) & \
                    (points[:, 1] > POINT_CLOUD_RANGE[1] + 1e-4) & (points[:, 1] < POINT_CLOUD_RANGE[4] - (1e-4)) & \
                    (points[:, 2] > POINT_CLOUD_RANGE[2] + 1e-4) & (points[:, 2] < POINT_CLOUD_RANGE[5] - (1e-4))
        if torch.sum(valid_ind.type(torch.float)) <= 0:
            return grid_3d.permute(2, 1, 0) + 2

        p_valid = points[valid_ind]

        grid_3d[((p_valid[:, 0] - POINT_CLOUD_RANGE[0] + 1e-5) / VOXEL_SIZE[0]).type(torch.long),
                ((p_valid[:, 1] - POINT_CLOUD_RANGE[1] + 1e-5) / VOXEL_SIZE[1]).type(torch.long),
                ((p_valid[:, 2] - POINT_CLOUD_RANGE[2] + 1e-5) / VOXEL_SIZE[2]).type(torch.long)] = 1


        grid_3d = grid_3d.permute(2, 1, 0)

        '''obtain grid label'''
        H, W, D = grid_3d.shape
        grid_label = torch.ones_like(grid_3d) * 2

        ind_h, ind_w, ind_d = torch.meshgrid([torch.arange(H), torch.arange(W), torch.arange(D)])

        index_full = torch.stack([ind_h+0.5, ind_w+0.5, ind_d+0.5], dim=3)
        valid_points_ind = index_full[grid_3d != 0]
        valid_num = valid_points_ind.shape[0]

        rate_h, rate_w = (valid_points_ind[:, 0] - center[0]) / (valid_points_ind[:, 2] - center[2] + 1e-6), \
                         (valid_points_ind[:, 1] - center[1]) / (valid_points_ind[:, 2] - center[2] + 1e-6)
        depth_points = torch.arange(D).unsqueeze(0).repeat(valid_num, 1)
        new_h, new_w = (depth_points - center[2]) * rate_h[:, None, ...] + center[0], \
                       (depth_points - center[2]) * rate_w[:, None, ...] + center[1]

        # depth_points_ind = depth_points <= valid_points_ind[:, 2:3]
        valid_index = (new_h < H) & (new_w < W) & (new_h > 0) & (new_w > 0)
        depth_points_ind = (depth_points <= valid_points_ind[:, 2:3]) & valid_index
        depth_points_ind_out = (depth_points > valid_points_ind[:, 2:3]) & valid_index

        grid_label[new_h[depth_points_ind].type(torch.long),
                   new_w[depth_points_ind].type(torch.long),
                   depth_points[depth_points_ind].type(torch.long)] = 0

        grid_label[new_h[depth_points_ind_out].type(torch.long),
                   new_w[depth_points_ind_out].type(torch.long),
                   depth_points[depth_points_ind_out].type(torch.long)] = 2
        grid_label[grid_3d != 0] = 1

        return grid_label



    def occupancy_loss_1213(self, occupancy_prob_0,
                            aug_lidar_coor_map, **kwargs):
        aug_lidar_3d = aug_lidar_coor_map.reshape(occupancy_prob_0.shape[0], -1, 3)
        occupancy_gt = torch.stack([self.occupancy_label_1213(p) for p in aug_lidar_3d], dim=0)

        cls_loss = self.focal_loss(occupancy_prob_0, occupancy_gt)
        tb_dict = {"occupancy_loss_1213": cls_loss.item()}
        return cls_loss, tb_dict


    def focal_loss(self, input, target, alpha=0.25, gamma=2.):
        '''
        Args:
            input:  prediction, 'batch x c x h x w'
            target:  ground truth, 'batch x c x h x w'
            alpha: hyper param, default in 0.25
            gamma: hyper param, default in 2.0
        Reference: Focal Loss for Dense Object Detection, ICCV'17
        '''

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        loss = 0

        pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds * alpha
        neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * (1 - alpha)

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss.mean()

    def get_loss(self):
        """
        Gets DDN loss
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """

        loss, tb_dict = self.ffn.get_loss()

        loss2, tb_dict2 = self.occupancy_loss(**self.forward_ret_dict)
        loss += loss2
        tb_dict.update(tb_dict2)

        loss3, tb_dict3 = self.occupancy_loss_1213(**self.forward_ret_dict)
        loss += loss3
        tb_dict.update(tb_dict3)


        return loss, tb_dict
