import numpy as np
import open3d as o3d
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pykitti


def load_kitti_scan(kitti, frame):
    """ Load a LiDAR scan from a KITTI dataset using pykitti. """
    points = kitti.get_velo(frame)[:, :2]  # Extract X, Y coordinates
    return points


def get_kitti_odometry(kitti, frame):
    """ Retrieve odometry pose for a given frame. """
    return kitti.poses[frame]


def create_occupancy_grid(points, grid_size=0.05, grid_extent=50):
    """ Convert a 2D point cloud into an occupancy grid. """
    res = int(grid_extent / grid_size)
    grid = np.zeros((res, res))

    # Normalize points to grid indices
    min_x, min_y = points.min(axis=0)
    points -= np.array([min_x, min_y])

    indices = (points / grid_size).astype(int)
    indices = np.clip(indices, 0, res - 1)

    for x, y in indices:
        grid[y, x] = 1  # Mark occupied cells

    return grid


def generate_probability_map(occupancy_grid, sigma=1.0):
    """ Apply Gaussian blur to create a probability map. """
    return ndi.gaussian_filter(occupancy_grid, sigma=sigma)


def compute_correlation(reference_map, query_map, search_range, step_size):
    """ Brute-force search over (dx, dy, theta) to find the best alignment. """
    best_score = -np.inf
    best_transform = None

    dx_range = np.arange(-search_range, search_range, step_size)
    dy_range = np.arange(-search_range, search_range, step_size)
    theta_range = np.linspace(-np.pi / 4, np.pi / 4, 10)  # Rotate in small steps

    for theta in theta_range:
        rotated_query = ndi.rotate(query_map, np.degrees(theta), reshape=False)
        for dx in dx_range:
            for dy in dy_range:
                shifted_query = ndi.shift(rotated_query, shift=[dy, dx])
                score = np.sum(shifted_query * reference_map)

                if score > best_score:
                    best_score = score
                    best_transform = (dx, dy, theta)

    return best_transform, best_score


# Example usage
basedir = "/Users/thibhero/dataset"
sequence = "00"

kitti = pykitti.odometry(basedir, sequence)

frame_ref, frame_query = 0, 10  # Example frames
ref_points = load_kitti_scan(kitti, frame_ref)
query_points = load_kitti_scan(kitti, frame_query)

ref_grid = create_occupancy_grid(ref_points)
query_grid = create_occupancy_grid(query_points)

ref_prob_map = generate_probability_map(ref_grid)
query_prob_map = generate_probability_map(query_grid)

best_transform, score = compute_correlation(ref_prob_map, query_prob_map, search_range=5, step_size=0.2)
print("Best Transform (dx, dy, theta):", best_transform)

# Get odometry for reference
odometry_ref = get_kitti_odometry(kitti, frame_ref)
odometry_query = get_kitti_odometry(kitti, frame_query)
print("Odometry Reference:", odometry_ref)
print("Odometry Query:", odometry_query)
