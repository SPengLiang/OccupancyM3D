from collections import OrderedDict
from pathlib import Path
from torch import hub

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kornia.enhance.normalize import normalize
except:
    pass
    # print('Warning: kornia is not installed. This package is only required by CaDDN')

from . import dla
from . import dlaup
import math

class DDNTemplate(nn.Module):

    def __init__(self, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            # constructor: function, Model constructor
            # feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        # self.num_classes = 2
        # self.num_classes = 1
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss

        if self.pretrained:
            # Preprocess Module
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])


        # self.backbone = dla.dla34(dd3d=False, pretrained=True, return_levels=True)
        # self.backbone = dla.dla34(dd3d=True, pretrained=True, return_levels=True)
        self.backbone = dla.dla34(dd3d=True, pretrained=True, pretrained_path=self.pretrained_path, return_levels=True)

        self.first_level = 2
        channels = self.backbone.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.feat_up = dlaup.DLAUp(channels[self.first_level:], scales_list=scales)
        self.depth_class = nn.Conv2d(64, 81, kernel_size=1)
        # self.depth_uncer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, images):
        """
        for DLA34

        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        # torch.cuda.empty_cache()

        # Preprocess images
        x = self.preprocess(images)

        max_h, max_w = 384, 1248
        cur_h, cur_w = images.shape[-2:]
        x = F.interpolate(x, size=(max_h, max_w), mode='bilinear', align_corners=True)
        feat_shape = math.ceil(cur_h / 4), math.ceil(cur_w / 4)

        # Extract features
        result = OrderedDict()
        x = self.backbone(x)   # 1/1, 1/2, 1/4, 1/8, 1/16
        result['features'] = F.interpolate(x[2], size=feat_shape, mode='bilinear', align_corners=True)
        x = self.feat_up(x[self.first_level:])

        x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=True)

        # Prediction classification logits
        logits = self.depth_class(x)
        result["logits"] = logits

        return result

    def gen_center_depths(self, depth_min=2, depth_max=46.8, num_bins=80, mode='LID'):
        ind = torch.arange(num_bins)
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            center_depth = bin_size * ind + depth_min
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            center_depth = ((1 + 2*ind)**2 - 1) * bin_size / 8 + depth_min
        else:
            raise NotImplementedError

        center_depth = torch.cat([center_depth, torch.tensor([depth_max+3]).to(center_depth.device)])
        return center_depth

    def guassin(self, shift, gamma=1.):
        import numpy as np
        coeff = 1 / (np.sqrt(np.pi * 2) * gamma)
        expon = - (shift ** 2 / (2 * gamma ** 2))
        prob = coeff * torch.exp(expon)
        relative_prob = prob / coeff

        return prob, relative_prob

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            # Create a mask for padded pixels
            mask = torch.isnan(x)

            # Match ResNet pretrained preprocessing
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            x[mask] = 0

        return x
