from functools import partial

import numpy as np

from ...utils import common_utils, calibration_kitti
from . import augmentor_utils, database_sampler

from numpy import random
import cv2
from ...utils import box_utils
from ..kitti import kitti_utils


def convert_to_3d(depth, P2, upsample_factor, x_start, y_start):
    fx = P2[0][0] * upsample_factor
    fy = P2[1][1] * upsample_factor
    cx = P2[0][2] * upsample_factor
    cy = P2[1][2] * upsample_factor

    b_x = P2[0][3] * upsample_factor / (-fx)
    b_y = P2[1][3] * upsample_factor / (-fy)
    # print(fx, fy, cx, cy)

    x_tile = np.array(range(depth.shape[1])).reshape(1, -1) + x_start
    points_x = np.tile(x_tile, [depth.shape[0], 1])

    y_tile = np.array(range(depth.shape[0])).reshape(-1, 1) + y_start
    points_y = np.tile(y_tile, [1, depth.shape[1]])

    points_x = points_x.reshape((-1, 1))
    points_y = points_y.reshape((-1, 1))
    depth = depth.reshape((-1, 1))

    # # -------mask-------
    # # mask = np.where(depth != np.inf, True, False)
    # mask = np.where(depth > 0, True, False)
    # points_x = points_x[mask].reshape((-1, 1))
    # points_y = points_y[mask].reshape((-1, 1))
    # depth = depth[mask].reshape((-1, 1))

    uv_points = np.concatenate([points_x, points_y], axis=1)

    points_x = (points_x - cx) / fx
    points_y = (points_y - cy) / fy

    points_x = points_x * depth + b_x
    points_y = points_y * depth + b_y

    points = np.concatenate([points_x, points_y, depth], axis=1)

    return points, uv_points

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    #scale all area
    # src_dir = get_dir([0, scale_tmp[1] * -0.5], rot_rad)
    # dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    #scale all area
    # src[2, :] = np.array([center[0] - 0.5 * scale_tmp[0], center[1] - 0.5 * scale_tmp[1]])
    # dst[2, :] = np.array([0, 0])

    if inv:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return trans, trans_inv
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


class DataAugmentorAffine(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        lidar_depth = data_dict["lidar_depth"]
        if "depth_maps" in data_dict.keys():
            depth_maps = data_dict["depth_maps"]
            for cur_axis in config['ALONG_AXIS_LIST']:
                assert cur_axis in ['horizontal']
                # images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                #     images, depth_maps, gt_boxes, calib,
                # )
                # images, depth_maps, gt_boxes, gt_boxes2d = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                #     images, depth_maps, gt_boxes, calib, gt_boxes2d
                # )
                images, depth_maps, gt_boxes, gt_boxes2d, lidar_depth, aug_lidar_coor_map = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                    images, depth_maps, gt_boxes, calib, gt_boxes2d, lidar_depth
                )
            data_dict['depth_maps'] = depth_maps
        else:
            for cur_axis in config['ALONG_AXIS_LIST']:
                assert cur_axis in ['horizontal']
                # images, _, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                #     images, np.zeros((1, 1)), gt_boxes, calib,
                # )
                # images, _, gt_boxes, gt_boxes2d = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                #     images, np.zeros((1, 1)), gt_boxes, calib, gt_boxes2d
                # )
                images, _, gt_boxes, gt_boxes2d, lidar_depth, aug_lidar_coor_map = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                    images, np.zeros((1, 1)), gt_boxes, calib, gt_boxes2d, lidar_depth
                )
        data_dict['images'] = images
        data_dict['gt_boxes'] = gt_boxes

        data_dict['gt_boxes2d'] = gt_boxes2d
        data_dict['lidar_depth'] = lidar_depth
        data_dict['aug_lidar_coor_map'] = aug_lidar_coor_map
        return data_dict

    def random_image_affine(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_affine, config=config)

        import copy

        images = copy.deepcopy(data_dict["images"])
        gt_boxes = copy.deepcopy(data_dict['gt_boxes'])
        gt_boxes2d = copy.deepcopy(data_dict["gt_boxes2d"])
        gt_boxes_mask = copy.deepcopy(data_dict['gt_boxes_mask'])
        calib = copy.deepcopy(data_dict["calib"])
        depth_maps = copy.deepcopy(data_dict["depth_maps"]) # dense
        lidar_depth = copy.deepcopy(data_dict["lidar_depth"]) # sparse

        random_flip = config['random_flip']
        random_crop = config['random_crop']
        scale = config['scale']
        shift = config['shift']
        # resolution = config['resolution']
        if images.shape[1] > 1500:
            resolution = config['resolution']
        else:
            resolution = np.array([images.shape[1], images.shape[0]])

        gt_boxes = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes, calib)
        # boxes3d_kitti_camera_to_lidar [x, y, z, l, h, w, r]


        img_size = np.array([images.shape[1], images.shape[0]])
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False

        if np.random.random() < random_flip:
            random_flip_flag = True
            images = cv2.flip(images, 1)
            depth_maps = cv2.flip(depth_maps, 1)
            lidar_depth = cv2.flip(lidar_depth, 1)

        if np.random.random() < random_crop:
            random_crop_flag = True
            crop_size = img_size * np.clip(np.random.randn()*scale + 1, 1 - scale, 1 + scale)
            center[0] += img_size[0] * np.clip(np.random.randn() * shift, -2 * shift, 2 * shift)
            center[1] += img_size[1] * np.clip(np.random.randn() * shift, -2 * shift, 2 * shift)
        depth_scale_factor = crop_size[1] / resolution[1]
        depth_maps = depth_maps * depth_scale_factor
        lidar_depth = lidar_depth * depth_scale_factor

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, resolution, inv=1)
        images = cv2.warpAffine(images, trans, tuple(resolution), flags=cv2.INTER_LINEAR)
        depth_maps = cv2.warpAffine(depth_maps, trans, tuple(resolution), flags=cv2.INTER_NEAREST)
        lidar_depth = cv2.warpAffine(lidar_depth, trans, tuple(resolution), flags=cv2.INTER_NEAREST)

        aug_lidar_3d, _ = convert_to_3d(lidar_depth, calib.P2, 1, 0, 0)
        # aug_lidar_3d, _ = convert_to_3d(depth_maps, calib.P2, 1, 0, 0)
        aug_lidar_3d = calib.rect_to_lidar(aug_lidar_3d)
        aug_lidar_coor_map = np.zeros((images.shape[0], images.shape[1], 3))
        aug_lidar_coor_map[..., 0] = aug_lidar_3d[:, 0].reshape(images.shape[0], images.shape[1])
        aug_lidar_coor_map[..., 1] = aug_lidar_3d[:, 1].reshape(images.shape[0], images.shape[1])
        aug_lidar_coor_map[..., 2] = aug_lidar_3d[:, 2].reshape(images.shape[0], images.shape[1])


        object_num = len(gt_boxes)
        if random_flip_flag:
            # calib.flip(img_size)
            for i in range(object_num):
                [x1, _, x2, _] = gt_boxes2d[i]
                gt_boxes2d[i, 0], gt_boxes2d[i, 2] = img_size[0] - x2, img_size[0] - x1
                gt_boxes[i, -1] = np.pi - gt_boxes[i, -1]
                # gt_boxes[i, 0] *= -1
                img_pts, img_depth = calib.rect_to_img(gt_boxes[i:i+1, :3])
                W = img_size[0]
                img_pts[:, 0] = W - img_pts[:, 0]
                pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
                # print(gt_boxes[i:i+1, :3], pts_rect, img_pts, img_depth)
                gt_boxes[i:i+1, :3] = pts_rect

                if gt_boxes[i, -1] > np.pi:  gt_boxes[i, -1] -= 2 * np.pi
                if gt_boxes[i, -1] < -np.pi: gt_boxes[i, -1] += 2 * np.pi

        for i in range(object_num):
            heading_angle = calib.ry2alpha(gt_boxes[i, -1], (gt_boxes2d[i, 0] + gt_boxes2d[i, 2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi

            # add affine transformation for 2d boxes.
            gt_boxes2d[i, :2] = affine_transform(gt_boxes2d[i, :2], trans)
            gt_boxes2d[i, 2:] = affine_transform(gt_boxes2d[i, 2:], trans)
            gt_boxes[i, -1] = calib.alpha2ry(heading_angle, (gt_boxes2d[i, 0] + gt_boxes2d[i, 2]) / 2)

            if gt_boxes2d[i, 2] < 10 or gt_boxes2d[i, 0] >= resolution[0]-10 or gt_boxes2d[i, 3] < 10 or gt_boxes2d[i, 1] >= resolution[1]-10:
                gt_boxes_mask[i] = False
                continue

            center_3d = gt_boxes[i, :3] + [0, -gt_boxes[i, 4] / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            new_loc_3d = np.zeros(3)
            new_loc_3d[2] = gt_boxes[i, 2] * depth_scale_factor
            new_loc_3d[0] = ((center_3d[0] - calib.P2[0, 2])*new_loc_3d[2] - calib.P2[0, 3]) / calib.P2[0, 0]
            new_loc_3d[1] = ((center_3d[1] - calib.P2[1, 2])*new_loc_3d[2] - calib.P2[1, 3]) / calib.P2[1, 1] + gt_boxes[i, 4] / 2
            # new_loc_3d[1] = ((center_3d[1] - calib.P2[1, 2])*new_loc_3d[2] - calib.P2[1, 3]) / calib.P2[1, 1]

            gt_boxes[i, :3] = new_loc_3d
            # gt_boxes[i, 3:6] *= depth_scale_factor

        gt_boxes = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes, calib)


        data_dict["images"] = images
        data_dict['gt_boxes'] = gt_boxes
        data_dict["gt_boxes2d"] = gt_boxes2d
        data_dict["gt_boxes_mask"] = gt_boxes_mask
        data_dict["depth_maps"] = depth_maps  # dense
        data_dict["lidar_depth"] = lidar_depth  # sparse
        data_dict["aug_lidar_coor_map"] = aug_lidar_coor_map
        # data_dict["trans_lidar_to_cam"], data_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        return data_dict


    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        if noise_translate_std == 0:
            return data_dict
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_translation_along_%s' % cur_axis)(
                gt_boxes, points, noise_translate_std,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper:
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                              config['SWAP_PROB'],
                                                              config['SWAP_MAX_NUM'],
                                                              pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')

        return data_dict
