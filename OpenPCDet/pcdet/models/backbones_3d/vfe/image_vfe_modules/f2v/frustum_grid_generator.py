import torch
import torch.nn as nn

try:
    from kornia.utils.grid import create_meshgrid3d
    from kornia.geometry.linalg import transform_points
except Exception as e:
    # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
    # print('Warning: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
    pass

from pcdet.utils import transform_utils

from datetime import datetime


class FrustumGridGenerator(nn.Module):

    def __init__(self, grid_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()
        try:
            import kornia
        except Exception as e:
            # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
            print('Error: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. '
                  'Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')
            exit(-1)

        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size, dtype=self.dtype)
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        # Create voxel grid
        self.depth, self.width, self.height = self.grid_size.int()
        self.voxel_grid = create_meshgrid3d(depth=self.depth,
                                                         height=self.height,
                                                         width=self.width,
                                                         normalized_coordinates=False)

        self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
        Returns:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)
        return unproject

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid: (B, X, Y, Z, 3), Voxel sampling grid
            grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
        Returns:
            frustum_grid: (B, X, Y, Z, 3), Frustum sampling grid
        """
        B = lidar_to_cam.shape[0]

        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        trans = C_V @ V_G

        # Reshape to match dimensions
        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        # Transform to camera frame
        camera_grid = transform_points(trans_01=trans, points_1=voxel_grid)

        # Project to image
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = transform_utils.project_to_image(project=I_C, points=camera_grid)

        # Convert depths to depth bins
        image_depths = transform_utils.bin_depths(depth_map=image_depths, **self.disc_cfg)

        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return frustum_grid

    def forward(self, lidar_to_cam, cam_to_img, image_shape):
    # def trans(self, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
            image_shape: (B, 2), Image shape [H, W]
        Returns:
            frustum_grid (B, X, Y, Z, 3), Sampling grids for frustum features
        """

        frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid.to(lidar_to_cam.device),
                                           grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
                                           lidar_to_cam=lidar_to_cam,
                                           cam_to_img=cam_to_img)

        # Normalize grid
        image_shape, _ = torch.max(image_shape, dim=0)
        image_depth = torch.tensor([self.disc_cfg["num_bins"]],
                                   device=image_shape.device,
                                   dtype=image_shape.dtype)
        frustum_shape = torch.cat((image_depth, image_shape))
        frustum_grid = transform_utils.normalize_coords(coords=frustum_grid, shape=frustum_shape)

        # Replace any NaNs or infinites with out of bounds
        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid




import numpy as np
class FrustumGridGenerator_Numpy():

    def __init__(self, grid_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()

        self.grid_size = grid_size
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        # Calculate voxel size
        pc_range = pc_range.reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        # Create voxel grid
        self.depth, self.width, self.height = self.grid_size.astype(np.int32)
        # self.voxel_grid = create_meshgrid3d(depth=self.depth,
        #                                                  height=self.height,
        #                                                  width=self.width,
        #                                                  normalized_codinates=False)

        # ind_d, ind_h, ind_w = np.meshgrid(np.arange(self.depth), np.arange(self.height), np.arange(self.width))
        ind_h, ind_d, ind_w = np.meshgrid(np.arange(self.height), np.arange(self.depth), np.arange(self.width))
        self.voxel_grid = np.stack([ind_d, ind_w, ind_h], axis=-1)[None, ...].astype(np.float32)


        self.voxel_grid = np.transpose(self.voxel_grid, [0, 1, 3, 2, 4])

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
        Returns:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = np.array([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=np.float32)  # (4, 4)

        return unproject

    def bin_depths_numpy(self, depth_map, mode, depth_min, depth_max, num_bins, target=False):
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
        import warnings
        warnings.filterwarnings('ignore')
        import math
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * np.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (np.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~np.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.astype(np.int64)
        return indices

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid: (B, X, Y, Z, 3), Voxel sampling grid
            grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
        Returns:
            frustum_grid: (B, X, Y, Z, 3), Frustum sampling grid
        """

        B = lidar_to_cam.shape[0]

        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        trans = C_V @ V_G
        # print(cam_to_img)

        # Reshape to match dimensions
        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat(repeats=B, axis=0)

        B, X, Y, Z, _ = voxel_grid.shape


        # Transform to camera frame
        a = datetime.now()
        # camera_grid = transform_points(trans_01=torch.tensor(trans), points_1=torch.tensor(voxel_grid))
        # voxel_grid_set = voxel_grid.reshape(B, -1, 3)
        # voxel_grid_set = np.concatenate([voxel_grid_set, np.ones_like(voxel_grid_set[..., :1])], axis=-1)
        # trans_set = trans.reshape(B, 4, 4)
        # camera_grid = np.stack([v_set @ t_set for v_set, t_set in zip(voxel_grid_set, trans_set)], axis=0)
        # camera_grid = camera_grid[..., :3] / camera_grid[..., 3:]
        # camera_grid = camera_grid.numpy().reshape(B, -1, 3)
        # print(camera_grid.shape)
        # print(camera_grid.shape, np.min(camera_grid), np.max(camera_grid), camera_grid[0, 500, :], voxel_grid_set[0, 500, :])

        trans_01, points_1 = trans, voxel_grid
        shape_inp = list(points_1.shape)
        points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
        trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
        trans_01 = trans_01.repeat(repeats=points_1.shape[0] // trans_01.shape[0], axis=0)

        points_1_h = np.concatenate([points_1, np.ones_like(points_1[..., :1])], axis=-1)
        points_0_h = points_1_h @ np.transpose(trans_01, (0, 2, 1))
        b = datetime.now()


        points_0 = points_0_h[..., :-1] / points_0_h[..., -1:]
        shape_inp[-2] = points_0.shape[-2]
        shape_inp[-1] = points_0.shape[-1]
        points_0 = points_0.reshape(shape_inp)
        camera_grid = points_0.reshape(B, -1, 3)
        c = datetime.now()

        # print('b-a', b-a, c-b)


        '''
        shape_inp = list(points_1.shape)
        points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
        trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
        # We expand trans_01 to match the dimensions needed for bmm
        trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)
        # to homogeneous
        points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
        # transform coordinates
        points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
        points_0_h = torch.squeeze(points_0_h, dim=-1)
        # to euclidean
        points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
        # reshape to the input shape
        shape_inp[-2] = points_0.shape[-2]
        shape_inp[-1] = points_0.shape[-1]
        points_0 = points_0.reshape(shape_inp)
        return points_0

        '''

        # Project to image
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        # image_grid, image_depths = transform_utils.project_to_image(project=torch.tensor(I_C), points=torch.tensor(camera_grid))
        camera_grid_set = np.concatenate([camera_grid, np.ones_like(camera_grid[..., :1])], axis=-1)
        I_C_set = I_C.reshape(B, 3, 4)
        proj_set = np.stack([(t_set @ (v_set.T)).T for t_set, v_set in zip(I_C_set, camera_grid_set)], axis=0)
        proj_set = proj_set.reshape((B, X, Y, Z, 3))
        image_depths = proj_set[..., 2]
        image_grid = proj_set[..., :2] / proj_set[..., 2:]
        # print(np.min(image_depths), np.max(image_depths))


        # Convert depths to depth bins
        image_depths = self.bin_depths_numpy(depth_map=image_depths, **self.disc_cfg)

        # Stack to form frustum grid
        frustum_grid = np.concatenate((image_grid, image_depths[..., None]), axis=-1)
        return frustum_grid

    def trans(self, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
            cam_to_img: (B, 3, 4), Camera projection matrix
            image_shape: (B, 2), Image shape [H, W]
        Returns:
            frustum_grid (B, X, Y, Z, 3), Sampling grids for frustum features
        """

        frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid,
                                           grid_to_lidar=self.grid_to_lidar,
                                           lidar_to_cam=lidar_to_cam,
                                           cam_to_img=cam_to_img)

        # Normalize grid
        image_shape = np.max(image_shape, axis=0)
        image_depth = np.array([self.disc_cfg["num_bins"]])
        frustum_shape = np.concatenate((image_depth, image_shape))

        def normalize_coords_numpy(coords, shape):
            """
            Normalize coordinates of a grid between [-1, 1]
            Args:
                coords: (..., 3), Coordinates in grid
                shape: (3), Grid shape
            Returns:
                norm_coords: (.., 3), Normalized coordinates in grid
            """
            min_n = -1
            max_n = 1
            shape = np.flip(shape, axis=[0])  # Reverse ordering of shape

            # Subtract 1 since pixel indexing from [0, shape - 1]
            norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
            return norm_coords

        frustum_grid = normalize_coords_numpy(coords=frustum_grid, shape=frustum_shape)


        # Replace any NaNs or infinites with out of bounds
        mask = ~np.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid

if __name__ == '__main__':
    POINT_CLOUD_RANGE = [2, -30.08, -3.0, 46.8, 30.08, 1.0]
    VOXEL_SIZE = [0.16, 0.16, 0.16]
    grid_size = (np.array(POINT_CLOUD_RANGE[3:6]) - np.array(POINT_CLOUD_RANGE[0:3])) / np.array(VOXEL_SIZE)
    grid_size = np.round(grid_size).astype(np.int64)
    disc_cfg = {
                "mode": 'LID',
                "num_bins": 80,
                "depth_min": 2.0,
                "depth_max": 46.8
                }

    grid_pt = FrustumGridGenerator(grid_size, np.array(POINT_CLOUD_RANGE), disc_cfg)
    grid_np = FrustumGridGenerator_Numpy(grid_size, np.array(POINT_CLOUD_RANGE), disc_cfg)


    trans_lidar_to_cam = np.array([[2.34773921e-04, -9.99944150e-01, -1.05634769e-02, -2.79681687e-03],
                                   [1.04494076e-02, 1.05653545e-02, -9.99889612e-01, -7.51087889e-02],
                                   [9.99945343e-01, 1.24365499e-04, 1.04513029e-02, -2.72132814e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], dtype=np.float32)
    trans_cam_to_img = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                                 [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                                 [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]], dtype=np.float32)
    image_shape = np.array([376, 1248], dtype=np.float32)



    a = datetime.now()
    r_pt = grid_pt(torch.tensor(trans_lidar_to_cam[None, ...]),
                   torch.tensor(trans_cam_to_img[None, ...]),
                   torch.tensor(image_shape[None, ...]))
    b = datetime.now()
    r_np = grid_np.trans(trans_lidar_to_cam[None, ...], trans_cam_to_img[None, ...], image_shape[None, ...])
    c = datetime.now()
    print('pt:', b-a, 'np:', c-b)


    diff = np.abs(r_pt.numpy() - r_np)
    # print(np.sum(diff > 0))
    print(r_pt.shape, r_np.shape)
    print(np.sum(np.abs(r_pt.numpy()-r_np)))
    c = 1
    pass
