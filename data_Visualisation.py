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
    plt.title('Map')
    plt.savefig(f"images/map/{id}")
    plt.close() # Close the figure to free up memory

def plot_Perf(historic):
    Cov = historic["cov"]
    tps = historic["t"]

    plt.figure()
    plt.subplot(311)
    plt.plot(tps, Cov[:,0], label="Vxx")
    plt.ylabel("temps (s)")
    plt.ylabel("Cov")
    plt.legend(loc="upper right")

    plt.subplot(312)
    plt.plot(tps, Cov[:,1],  label="Vxy")
    plt.ylabel("temps (s)")
    plt.ylabel("Cov")
    plt.legend(loc="upper right")

    plt.subplot(313)
    plt.plot(tps, Cov[:,2],  label="Vyy")
    plt.ylabel("temps (s)")
    plt.ylabel("Cov")
    plt.legend(loc="upper right")

    plt.suptitle("Variance sur la postion")
    plt.savefig(f"images/performances")
    plt.close() # Close the figure to free up memory

def plot_TreeCov(treesCov):
    temps = treesCov["t"]
    T1 = treesCov["1"]
    T2 = treesCov["2"]
    T3 = treesCov["3"]
    T4 = treesCov["4"]
    T5 = treesCov["5"]
    plt.figure()
    plt.plot(temps, T1[:,0], label="tree_50")
    plt.plot(temps, T1[:,1])
    plt.plot(temps, T2[:,0], label="tree_150")
    plt.plot(temps, T2[:,1])
    plt.plot(temps, T3[:,0], label="tree_200")
    plt.plot(temps, T3[:,1])
    plt.plot(temps, T4[:,0], label="tree_300")
    plt.plot(temps, T4[:,1])
    plt.plot(temps, T5[:,0], label="tree_500")
    plt.plot(temps, T5[:,1])

    plt.title("Norme varriance sur les landmarks")
    plt.ylabel("temps (s)")
    plt.ylabel("deviation standard")
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
  print(lower_bound)
  upper_bound = chi2.ppf((1 + confidence_level) / 2, degrees_of_freedom)
  within_bounds = np.logical_and(nis_values >= lower_bound, nis_values <= upper_bound)
  percentage_within_bounds = np.mean(within_bounds) * 100
  return lower_bound, upper_bound, percentage_within_bounds

def NIS_test(NIS_val):
    nis_values = NIS_val
    print(nis_values)
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

    plt.figure()
    plt.plot(nis_values)
    p1 = [0, len(nis_values)]
    p2 = [lower, lower]
    plt.plot(p1,p2)
    p1 = [0, len(nis_values)]
    p2 = [upper, upper]
    plt.plot(p1,p2)
    plt.show()

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


if __name__ == "__main__":
    loaded_nis = np.load('NIS_history.npy', allow_pickle=True)
    print(loaded_nis)
    NIS_test(loaded_nis)