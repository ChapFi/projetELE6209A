import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm
import copy
import open3d as o3d
import numpy as np


class EnhancedFrontend:
    def __init__(self, voxel_size=0.1, submap_size=10):
        self.voxel_size = voxel_size
        self.submap_size = submap_size  # Number of frames per submap
        self.current_submap = o3d.geometry.PointCloud()
        self.submaps = []
        self.submap_poses = []
        self.frame_count = 0
        self.current_pose = np.eye(4)
        self.poses = [np.eye(4)]  # Store all poses
        self.loop_closures = []  # Store loop closure constraints (i, j, transformation)

    def preprocess_pointcloud(self, points):
        """Preprocess point cloud by downsampling and computing normals"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Add intensity as colors if available (for KITTI)
        if points.shape[1] >= 4:
            intensities = points[:, 3]
            # Normalize and create grayscale color
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities) + 1e-8)
            colors = np.zeros((len(intensities), 3))
            colors[:, 0] = intensities
            colors[:, 1] = intensities
            colors[:, 2] = intensities
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Downsample with voxel grid
        pcd = pcd.voxel_down_sample(self.voxel_size)

        # Estimate normals (important for point-to-plane ICP)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

        # Orient normals consistently
        pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

        return pcd

    def extract_features(self, pcd, radius_normal=0.3, radius_feature=0.5):
        """Extract FPFH features from point cloud"""
        pcd_down = pcd.voxel_down_sample(radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal * 2, max_nn=30))

        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, fpfh

    def initial_alignment(self, source, target):
        """Global registration using FPFH features"""
        source_down, source_fpfh = self.extract_features(source)
        target_down, target_fpfh = self.extract_features(target)

        # RANSAC-based global registration
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.voxel_size * 1.5)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        return result.transformation, result.fitness

    def icp_refinement(self, source, target, init_transform=np.eye(4), max_distance=0.3):
        """Refined registration using Point-to-Plane ICP"""
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_distance, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=100)
        )

        return result.transformation, result.fitness, result.inlier_rmse

    def compute_information_matrix(self, source, target, transform, max_distance=0.3):
        """Compute information matrix based on matching quality"""
        # Match points
        # source_transformed = copy.deepcopy(source)
        # source_transformed.transform(transform)

        # Use correspondence distances to compute information matrix
        # dists = np.asarray(source.compute_point_cloud_distance(target))
        # print(dist)
        # inlier_dists = dists[dists < max_distance]
        #
        # if len(inlier_dists) == 0:
        #     # Fallback to identity with low confidence
        #     return np.eye(6) * 0.1
        #
        # # Compute precision based on inlier residuals
        # precision = 1.0 / (np.mean(inlier_dists) + 1)
        #
        # # Create information matrix (diagonal for simplicity)
        # # Higher weight for rotation, lower for translation
        # information = np.eye(6)
        # information[0:3, 0:3] *= precision  # Translation part
        # information[3:6, 3:6] *= precision * 10  # Rotation part (higher weight)

        _, cov_source = source.compute_mean_and_covariance()
        _, cov_target = target.compute_mean_and_covariance()

        return np.linalg.inv(cov_target)

    def detect_loop_closures(self, current_pcd, current_pose):
        """Detect loop closures with previous submaps"""
        loop_closures = []

        # Skip if we don't have enough submaps
        if len(self.submaps) < 2:
            return loop_closures

        # Only check against submaps that are far enough (skip recent ones)
        candidates = []
        for i, (submap, pose) in enumerate(zip(self.submaps, self.submap_poses)):
            # Skip the most recent submap
            if i >= len(self.submaps) - 1:
                continue

            # Compute distance between poses (simple Euclidean for translation part)
            distance = np.linalg.norm(current_pose[:3, 3] - pose[:3, 3])

            # If distance is small but not consecutive in time, it's a loop closure candidate
            if 5.0 < distance < 20.0:  # Reasonable distance range for loop closure
                candidates.append((i, distance, submap, pose))

        # Sort candidates by distance (ascending)
        candidates.sort(key=lambda x: x[1])

        # Check the top candidates
        for i, distance, submap, submap_pose in candidates[:3]:  # Check top 3
            # Initial alignment based on current estimate
            init_transform = np.linalg.inv(submap_pose) @ current_pose

            # Try to align with ICP
            transform, fitness, rmse = self.icp_refinement(
                current_pcd, submap, init_transform, max_distance=1.0)

            # If good match, confirm loop closure
            if fitness > 0.5 and rmse < 0.5:  # Thresholds to tune
                relative_pose = np.linalg.inv(submap_pose) @ transform @ current_pose
                information = self.compute_information_matrix(current_pcd, submap, transform)
                loop_closures.append((i, self.frame_count, relative_pose, information))
                print(f"Loop closure detected between submap {i} and current frame {self.frame_count}")

                # We'll take just the best loop closure
                break

        return loop_closures

    def process_frame(self, points):
        """Process a new LiDAR scan"""
        self.frame_count += 1

        # Preprocess the raw point cloud
        current_pcd = self.preprocess_pointcloud(points)

        # If first frame, just store it
        if self.frame_count == 1:
            self.current_submap = current_pcd
            return self.current_pose, None

        # Get the previous pointcloud (either last frame or submap)
        if (self.frame_count - 1) % self.submap_size == 0:
            # Using last submap
            reference = self.submaps[-1]
        else:
            # Using current submap
            reference = self.current_submap

        # Multi-stage alignment
        # 1. Initial alignment if frames are far apart
        init_transform, init_fitness = self.initial_alignment(current_pcd, reference)
        if init_fitness < 0.3:  # Poor alignment, fall back to odometry estimate
            init_transform = self.current_pose

        # 2. Refine with ICP
        transform, fitness, rmse = self.icp_refinement(
            current_pcd, reference, init_transform)

        # Update current pose (relative to previous pose)
        self.current_pose = np.linalg.inv(self.poses[-1]) @ self.current_pose @ transform
        self.poses.append(self.current_pose)

        # Add to current submap
        transformed_pcd = copy.deepcopy(current_pcd)
        transformed_pcd.transform(self.current_pose)

        if len(self.current_submap.points) == 0:
            self.current_submap = transformed_pcd.voxel_down_sample(self.voxel_size)
        else:
            self.current_submap += transformed_pcd
            # Down-sample the combined point cloud

        # Check if we should create a new submap
        loop_info = None
        if self.frame_count % self.submap_size == 0:
            # Store current submap
            self.submaps.append(self.current_submap)
            self.submap_poses.append(self.current_pose)

            # Detect loop closures
            loop_closures = self.detect_loop_closures(current_pcd, self.current_pose)

            if loop_closures:
                loop_info = loop_closures[0]  # Just use the first one
                self.loop_closures.append(loop_info)

            # Reset current submap
            self.current_submap = o3d.geometry.PointCloud()

        # Compute information matrix for odometry
        odom_info = self.compute_information_matrix(
            current_pcd, reference, self.current_pose)

        return self.current_pose, (self.frame_count - 2, self.frame_count - 1, transform, odom_info, loop_info)
