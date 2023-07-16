import torch
import torch.nn as nn


from .balancer import Balancer
from pcdet.utils import transform_utils

try:
    from kornia.losses.focal import FocalLoss
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class DDNLoss(nn.Module):

    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 disc_cfg,
                 fg_weight,
                 bg_weight,
                 downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        # self.balancer = Balancer(downsample_factor=downsample_factor,

        self.balancer = Balancer(downsample_factor=4,
        # self.balancer = Balancer(downsample_factor=2,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        # self.balancer = Balancer_with_uncer(downsample_factor=4,
        #                                      fg_weight=fg_weight,
        #                                      bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight

    def bin_ray_depths(self, depth_map, mode, depth_min, depth_max, num_bins, target=False):
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
        gt_one_hot = torch.zeros((B, num_bins+1, H, W), device=indices.device)
        bin_ind = torch.arange(num_bins+1, device=indices.device).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W)
        indices_repeat = indices.unsqueeze(1).repeat(1, num_bins+1, 1, 1)
        gt_one_hot[bin_ind == indices_repeat] = 1.
        gt_one_hot[bin_ind > indices_repeat] = 2.
        # gt_one_hot_v2 = torch.nn.functional.one_hot(indices, num_bins+1)
        # print(gt_one_hot.shape, gt_one_hot_v2.shape)
        # print('===============', torch.sum((gt_one_hot - gt_one_hot_v2.permute(0, 3, 1, 2))**2))

        return gt_one_hot


    def forward(self, depth_logits, depth_maps, gt_boxes2d):
    # def forward(self, depth_logits, depth_maps, gt_boxes2d, depth_uncer):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}

        # print(depth_logits.shape, depth_maps.shape)

        # Bin depth map to create target
        depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)

        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)

        # depth_target = self.bin_ray_depths(depth_maps, **self.disc_cfg, target=True)
        # depth_prob = torch.sigmoid(depth_logits)
        # mask = (depth_target < 2).type(torch.float)
        # loss = torch.nn.functional.binary_cross_entropy(depth_prob, depth_target, reduction='none')
        # # import pdb
        # # pdb.set_trace()
        # loss = torch.sum(loss * mask, dim=1) / torch.sum(mask, dim=1)
        # # print(torch.sum(loss), torch.sum(mask), torch.min(depth_target), torch.max(depth_target))

        # Compute foreground/background balancing
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)
        # loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d, log_variance=depth_uncer)

        # Final loss
        loss *= self.weight
        tb_dict.update({"ddn_loss": loss.item()})

        return loss, tb_dict