import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def visualize_matrix_sns(matrix, nb, id):
    sns.heatmap(matrix[:3+2*nb, :3+2*nb], cmap='viridis', annot=False, xticklabels=False, yticklabels=False) # annot=True to display values in each cell
    plt.title('Matrix corelation')
    plt.savefig(f"images/matrix/{id}")
    plt.close() # Close the figure to free up memory

def visualize_trajectorie(posMat, id):
    plt.figure()
    if posMat.shape == (3,):
        plt.plot(posMat[0], posMat[1])
    else:
        plt.plot(posMat[:,0], posMat[:,1])
    plt.title('SLAM path')
    plt.xlabel("longitude (m)")
    plt.ylabel("latitude (m)")
    plt.grid("on")
    plt.savefig(f"images/map/{id}")
    plt.close() # Close the figure to free up memory

def plot_Perf(historic):
    Cov = historic["cov"]
    tps = historic["t"]

    plt.figure()
    plt.subplot(311)
    plt.plot(tps, Cov[:,0], label="Vxx")
    plt.ylabel("Cov")
    plt.legend(loc="upper right")

    plt.subplot(312)
    plt.plot(tps, Cov[:,1],  label="Vyy")
    plt.ylabel("Cov")
    plt.legend(loc="upper right")

    plt.subplot(313)
    plt.plot(tps, Cov[:,2],  label="Vphi")
    plt.xlabel("time step")
    plt.ylabel("Cov")
    plt.legend(loc="upper right")

    plt.suptitle("Standard deviations on postion states")
    plt.savefig(f"images/performances.png")
    plt.close() # Close the figure to free up memory

def plot_TreeCov(treesCov):
    temps = treesCov["t"]
    T1 = treesCov["1"]
    T2 = treesCov["2"]
    T3 = treesCov["3"]
    T4 = treesCov["4"]
    T5 = treesCov["5"]
    plt.figure()
    plt.plot(temps, T1[:,0], label="tree_50", color="#9A0EEA")
    plt.plot(temps, T1[:,1], color="#9A0EEA")
    plt.plot(temps, T2[:,0], label="tree_150", color="#FE420F")
    plt.plot(temps, T2[:,1], color="#FE420F")
    plt.plot(temps, T3[:,0], label="tree_200", color="#FFD700")
    plt.plot(temps, T3[:,1], color="#FFD700")
    plt.plot(temps, T4[:,0], label="tree_300", color="#15B01A")
    plt.plot(temps, T4[:,1], color="#15B01A")
    plt.plot(temps, T5[:,0], label="tree_500", color="#13EAC9")
    plt.plot(temps, T5[:,1], color="#13EAC9")

    plt.title("Variances on landmarks")
    plt.xlabel("step time")
    plt.ylabel("standard deviation ")
    plt.legend(loc="upper right")
    plt.savefig(f"images/errDynamique")
    plt.close() # Close the figure to free up memory


def calculate_nis(innovation, innovation_covariance):
  """Calculates the Normalized Innovation Squared (NIS)."""
  return innovation.T @ np.linalg.inv(innovation_covariance) @ innovation

def check_nis_consistency(nis_values, measurement_dimension, confidence_level=0.95):
  """Checks the consistency of the NIS values using a chi-squared test.

  Args:
    nis_values (np.ndarray): Array of NIS values over time.
    measurement_dimension (int): Dimension of the measurement vector.
    confidence_level (float): Desired confidence level (e.g., 0.95 for 95%).

  Returns:
    tuple: (lower_bound, upper_bound, percentage_within_bounds)
  """
  degrees_of_freedom = measurement_dimension
  lower_bound = chi2.ppf((1 - confidence_level) / 2, degrees_of_freedom)
  upper_bound = chi2.ppf((1 + confidence_level) / 2, degrees_of_freedom)
  within_bounds = np.logical_and(nis_values >= lower_bound, nis_values <= upper_bound)
  percentage_within_bounds = np.mean(within_bounds) * 100
  return lower_bound, upper_bound, percentage_within_bounds

def NIS_test(NIS_val):
    nis_values = NIS_val
    measurement_dim = 2 # Get the dimension of the measurement vector
    lower, upper, percentage = check_nis_consistency(nis_values, measurement_dim)
    print(f"NIS Consistency Check (Confidence Level: 95%):")
    print(f"  Chi-squared distribution with {measurement_dim} degrees of freedom.")
    print(f"  Expected range: [{lower:.2f}, {upper:.2f}]")
    print(f"  Percentage of NIS values within the expected range: {percentage:.2f}%")

    if percentage < 95: # Or a threshold you deem appropriate
        print("Warning: The NIS test suggests potential inconsistency in the EKF.")
    else:
        print("The NIS test suggests the EKF is likely consistent.")

    # You can also analyze the average NIS over a window
    window_size = 50
    if len(nis_values) >= window_size:
        avg_nis = np.mean(nis_values[-window_size:])
        avg_lower = chi2.ppf(0.025, window_size * measurement_dim)
        avg_upper = chi2.ppf(0.975, window_size * measurement_dim)
        print(f"\nAverage NIS over the last {window_size} steps: {avg_nis:.2f}")
        print(f"Expected range for average NIS: [{avg_lower:.2f}, {avg_upper:.2f}]")
        if avg_nis < avg_lower or avg_nis > avg_upper:
            print("Warning: Average NIS suggests potential inconsistency.")
        else:
            print("Average NIS suggests consistency.")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(nis_values)
    p1 = [0, len(nis_values)]
    p2 = [lower, lower]
    ax.plot(p1,p2)
    p1 = [0, len(nis_values)]
    p2 = [upper, upper]
    ax.plot(p1,p2)
    textstr = f"percent in bounds: {percentage:.2f}%"
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
    plt.title('NIS test with 0.95 confidence level')
    plt.savefig("images/NIS_test.png")
    plt.close()

def plot_Map(state, landmarks, sigma):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(state[:, 0], state[:, 1], '-k', label='EKF Path')
    ax.set_xlabel('longitude (m)')
    ax.set_ylabel('lattitude (m)')
    ax.set_title('SLAM Landmarks with 95% Covariance Ellipses')
    ax.legend()
    plot_landmarks_with_cov(
        landmarks, sigma, ax=ax,
        n_std=2,
        edgecolor='blue', facecolor='none', linewidth=1, alpha=0.7
    )
    plt.savefig('images/finalMap.png')
    plt.close()

def plot_Time(timeHist):
    
    fig, ax = plt.subplots()
    plt.plot(timeHist["time"], timeHist["step"])
    plt.title("Computation time for steps")
    plt.xlabel("step time")
    plt.ylabel("computation time (s)")
    
    textstr = f"total time : {int(np.sum(timeHist["step"]))}s"
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)
    
    plt.savefig('images/computeTime.png')
    plt.close()

def find_circle_center_least_squares(points):
    """
    Calculates the center of a circle using least squares fitting.

    Args:
        points: A list of tuples, each representing a point (x, y).

    Returns:
        A tuple (h, k) representing the center of the circle.
    """
    n = len(points)
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    A = np.column_stack((-2 * x, -2 * y, np.ones(n)))
    b = -(x**2 + y**2)

    h, k, _ = np.linalg.lstsq(A, b, rcond=None)[0]

    return h, k

def displayRowData(rowOdom, rowLaser, sensorManager):
    X = np.array([[0],[0], [0]])
    Odom = np.zeros((3, len(rowOdom)+1))
    landMark = np.zeros((2,17233))
    landtree = np.zeros((2,17233))
    i = 0
    j = 0
    time =0
    for sensor in sensorManager:
        if sensor['sensor'] == 2:
            id = int(sensor['index'])
            mesure = rowOdom[id]
            dt = mesure['time_vs']-time
            time = dt + time
            vt = mesure['velocity']
            wt = mesure['steering']
            theta = X[2,0]
            Dmu = np.array([[vt*(-np.sin(theta)+np.sin(theta+wt*dt))/wt],
                        [vt*(np.cos(theta)-np.cos(theta+wt*dt))/wt],[wt*dt]])
            X = np.add(X, Dmu)
            Odom[:,i] = X[:,0]
            i += 1
        elif sensor['sensor'] == 3:
            id = int(sensor['index'])
            mesures = rowLaser[id]
            mesures = mesures['laser_values']
            angles = np.linspace(-np.pi/2, np.pi/2, 361)
            ranges =[]
            current_range =[]
            for angle, range in zip(angles, mesures):
                if range < 81:
                    current_range.append((angle,range))
                else:
                    if current_range:  # If there are measurements in the current range
                        ranges.append(current_range)
                        current_range = []  # Reset for the next range

            if current_range: #capture the last range if it doesn't end with the out of range value
                ranges.append(current_range)

            for dataRange in ranges:
                if len(dataRange) >= 8:

                    points = [(X[0,0]+pair[1]*np.cos(pair[0]+X[2,0]), X[1,0]+pair[1]*np.sin(pair[0]+X[2,0])) for pair in dataRange]
                    deltaB = dataRange[len(dataRange)-1][0]-dataRange[0][0]
                    (h,k) = find_circle_center_least_squares(points)
                    landtree[0,j] = h
                    landtree[1,j] = k

                    nbRange = len(dataRange)
                    angle = sum([pair[0] for pair in dataRange])/nbRange
                    range = sum([pair[1] for pair in dataRange])/nbRange
                    lm = np.array([[X[0,0]+range*np.cos(angle+X[2,0])],
                                            [X[1,0]+range*np.sin(angle+X[2,0])]])
                    landMark[:,j] = lm[:,0]
                    j += 1
    print(j)

    # Plot trajectory and landmark
    plt.figure()
    plt.scatter(landMark[0, :], landMark[1, :], marker='+', label='landmarks')
    plt.scatter(landtree[0, :], landtree[1, :], marker='.', label='landmarks')
    plt.plot(Odom[0, :], Odom[1,:], label='row trajectory', linestyle='--', color='r')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.title('EKF SLAM')
    plt.grid()
    plt.show()

def plot_landmarks_with_cov(landmarks, sigma, ax=None, n_std=2.0, **ellipse_kwargs):
    """
    Plot landmarks and their covariance ellipses.

    landmarks : your Landmarks() instance
    sigma     : full (3+2n)x(3+2n) covariance matrix
    ax        : optional matplotlib Axes
    n_std     : number of standard deviations for the ellipse (e.g. 2 = 95% confidence)
    ellipse_kwargs : passed to Ellipse (edgecolor, facecolor='none', alpha, etc.)
    """
    ax = ax or plt.gca()

    for j, lm in enumerate(landmarks.landmarks):
        # 1) extract mean
        cx, cy = lm.center

        # 2) extract 2x2 covariance for this landmark
        iL = 3 + 2*j
        cov = sigma[iL:iL+2, iL:iL+2]

        # 3) eigen‐decompose to get axis lengths & orientation
        vals, vecs = np.linalg.eigh(cov)
        # sort descending so vals[0] is largest
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]
        # angle of largest eigenvector
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        # width & height of ellipse = 2 * n_std * sqrt(eigenvalue)
        width, height = 2 * n_std * np.sqrt(vals)

        # 4) create & add the ellipse patch
        ell = Ellipse((cx, cy), width, height, angle=angle, **ellipse_kwargs)
        ax.add_patch(ell)

        # 5) plot center point
        ax.plot(cx, cy, 'ro', markersize=3)

    return ax

# final_state, robot_hist, lm_centers_hist, sigma_hist = \
#     EKFSlam(drsData, laserData, dataManagement)
#
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_aspect('equal', 'box')
#
# # line for robot path
# traj_line, = ax.plot([], [], '-k', lw=1)
#
# # scatter for landmark centers
# land_scat = ax.scatter([], [], s=20, c='tab:blue')
    #
    # # store ellipse artists
    # ellipses = []
    #
    # # autoscale
    # xs = robot_hist[:, 0]
    # ys = robot_hist[:, 1]
    # pad = 5
    # ax.set_xlim(xs.min() - pad, xs.max() + pad)
    # ax.set_ylim(ys.min() - pad, ys.max() + pad)
    #
    #
    # def animate(i):
    #     # 1) robot trajectory so far
    #     traj_line.set_data(robot_hist[:i + 1, 0], robot_hist[:i + 1, 1])
    #
    #     # 2) current landmark centers
    #     centers = lm_centers_hist[i]
    #     if centers:
    #         land_scat.set_offsets(centers)
    #     else:
    #         land_scat.set_offsets([])
    #
    #     # 3) draw covariance ellipses at this step
    #     #    first remove old
    #     for e in ellipses:
    #         e.remove()
    #     ellipses.clear()
    #
    #     sigma = sigma_hist[i]
    #     for j, (cx, cy) in enumerate(centers):
    #         # extract 2×2 sub‑cov for landmark j
    #         iL = 3 + 2 * j
    #         cov = sigma[iL:iL + 2, iL:iL + 2]
    #
    #         # eigen‑decompose
    #         vals, vecs = np.linalg.eigh(cov)
    #         order = vals.argsort()[::-1]
    #         vals, vecs = vals[order], vecs[:, order]
    #         angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    #         width, height = 2 * 2 * np.sqrt(vals)  # 2‑sigma ellipse
    #
    #         e = Ellipse((cx, cy), width, height, angle=angle,
    #                     edgecolor='C1', facecolor='none', lw=1, alpha=0.6)
    #         ax.add_patch(e)
    #         ellipses.append(e)
    #
    #     return [traj_line, land_scat] + ellipses
    #
    #
    # pbar = tqdm(total=len(robot_hist), desc="Saving animation")
    #
    #
    # # Define callback to update the progress bar
    # def progress_callback(frame_number, total_frames):
    #     pbar.update(1)
    #
    # ani = FuncAnimation(fig, animate,
    #                     frames=len(robot_hist),
    #                     interval=100)
    # ani.save("ekf.mp4", writer='ffmpeg', fps=60, dpi=200, bitrate=1800, progress_callback=progress_callback)
    # pbar.close()

if __name__ == "__main__":
    loaded_nis = np.load('images/NIS_history.npy', allow_pickle=True)
    NIS_test(loaded_nis)