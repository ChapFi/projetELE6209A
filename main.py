import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pykitti
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm
import copy
from Frontend import EnhancedFrontend
from GraphSLAM import GraphSLAM



def plot_trajectories(optimized_poses):
    """Compare true KITTI trajectory vs. SLAM trajectory."""
    opt_x, opt_y = [], []
    for key, T in optimized_poses.items():
        opt_x.append(float(T[0]))
        opt_y.append(float(T[1]))

    plt.figure(figsize=(10, 5))
    plt.plot(opt_x, opt_y, 'b-', label="LiDAR SLAM")
    plt.legend()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("KITTI LiDAR SLAM")
    plt.show()


def integrate_with_graphslam(dataset, graph, num_frames=None, frontend=None):
    """Integrate enhanced frontend with GraphSLAM"""
    if frontend is None:
        frontend = EnhancedFrontend(voxel_size=0.5, submap_size=10)

    # Reset frontend state
    frontend.frame_count = 0
    frontend.current_pose = np.eye(4)
    frontend.poses = [np.eye(4)]
    frontend.submaps = []
    frontend.submap_poses = []
    frontend.current_submap = o3d.geometry.PointCloud()

    num_frames = len(dataset) if num_frames is None else min(num_frames, len(dataset))

    # Add initial pose to graph
    graph.add_pose(0, np.array([0, 0, 0]))  # Start at origin

    for i in range(1, num_frames):
        # Get LiDAR scan from dataset
        current_scan = dataset.get_velo(i)

        # Process with enhanced frontend
        current_pose, constraints = frontend.process_frame(current_scan)

        # Debug print to track pose values
        print(f"Frame {i}, Pose: {current_pose[:3, 3]}, 2D Pose: {get_2d_pose(current_pose)}")

        # Add current pose to graph
        idx = frontend.frame_count - 1
        two_d_pose = get_2d_pose(current_pose)
        print(f"Adding node {idx} with pose {two_d_pose}")
        graph.add_pose(idx, two_d_pose)

        # Add odometry edge if we have constraints
        if constraints:
            prev_idx, curr_idx, transform, info_matrix, loop_info = constraints

            # Create edge with appropriate information matrix
            odom_info = np.eye(3)  # Simple identity for odometry
            graph.add_edge(prev_idx, curr_idx, get_2d_pose(transform), info_matrix)

            # Add loop closure if detected
            if loop_info:
                loop_from, loop_to, loop_transform, loop_info_matrix = loop_info
                # Convert 6D info matrix to 3D and add higher weight
                loop_info_3d = np.eye(3) * 10.0  # Higher weight for loop closures
                graph.add_edge(loop_from, loop_to, get_2d_pose(loop_transform), loop_info_matrix)

    return graph, frontend


def get_2d_pose(pose_matrix):
    """Extract 2D pose (x, y, theta) from SE(3) transformation matrix"""
    # Make sure scale is in meters
    x, y = pose_matrix[0, 3], pose_matrix[1, 3]

    # Proper extraction of yaw angle
    rot = pose_matrix[:3, :3]
    theta = np.arctan2(rot[1, 0], rot[0, 0])

    return np.array([x, y, theta])


def convert_information_matrix(info_matrix_6d):
    """Convert 6D SE(3) information matrix to 3D SE(2) information matrix"""
    # Extract parts relevant to x, y, and theta (typically indices 0, 1, and 5)
    info_3d = np.zeros((3, 3))
    info_3d[0, 0] = info_matrix_6d[0, 0]  # x-x
    info_3d[0, 1] = info_matrix_6d[0, 1]  # x-y
    info_3d[1, 0] = info_matrix_6d[1, 0]  # y-x
    info_3d[1, 1] = info_matrix_6d[1, 1]  # y-y
    info_3d[2, 2] = info_matrix_6d[5, 5]  # theta-theta

    return info_3d


# Load KITTI dataset
basedir = "/Users/thibhero/dataset"
sequence = "00"

if __name__ == "__main__":
    graph = GraphSLAM()
    frontend = EnhancedFrontend(voxel_size=0.5, submap_size=10)

    dataset = pykitti.odometry(basedir, sequence, frames=range(0, 4000, 100))
    graph, frontend = integrate_with_graphslam(dataset, graph, frontend=frontend)

    # Print initial poses before optimization
    print("Before optimization:")
    initial_poses = graph.get_nodes()

    # Plot initial trajectory
    init_x, init_y = [], []
    for key in sorted(initial_poses.keys()):
        pose = initial_poses[key]
        init_x.append(float(pose[0]))
        init_y.append(float(pose[1]))

    plt.figure(figsize=(10, 5))
    plt.plot(init_x, init_y, 'r-', label="Before Optimization")
    plt.title("Initial Trajectory")
    plt.show()

    print("Optimizing graph...")
    graph.optimize(max_iterations=100)

    print("After optimization:")
    print(graph.get_nodes())

    # Plot ground truth
    plt.figure(figsize=(10, 5))
    gt_x, gt_y = [], []
    for T in dataset.poses:
        gt_x.append(float(T[0, 3]))
        gt_y.append(float(T[1, 3]))
    plt.plot(gt_x, gt_y, 'g-', label="Ground Truth")
    plt.title("Ground Truth Trajectory")
    plt.show()

    plot_trajectories(graph.get_nodes())
