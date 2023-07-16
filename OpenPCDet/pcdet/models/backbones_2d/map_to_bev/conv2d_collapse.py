import torch
import torch.nn as nn

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg.ARGS)
        # # self.block = BasicBlock2D(in_channels=3 * self.num_heights,
        # self.block = BasicBlock2D(in_channels=32 * self.num_heights,
        # # self.block = BasicBlock2D(in_channels=8 * self.num_heights,
        #                           out_channels=self.num_bev_features,
        #                           **self.model_cfg.ARGS)

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        voxel_features = batch_dict["voxel_features"]
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)

        # import cv2 as cv
        # import numpy as np
        # cv.imwrite('/pvc_user/pengliang/MonoVoxel/tmp.png', (bev_features[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # exit(0)


        # b, _, h, w = bev_features.shape
        # coord_h, coord_w = torch.meshgrid([torch.arange(h), torch.arange(w)])
        # coord_h = coord_h.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1).cuda().type(torch.float32).contiguous()
        # coord_w = coord_w.unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1).cuda().type(torch.float32).contiguous()
        # # coord_h, coord_w = coord_h/h, coord_w/w
        # bev_features = torch.cat([bev_features[:, :-2, ...], coord_h, coord_w], dim=1)


        batch_dict["spatial_features"] = bev_features
        return batch_dict
