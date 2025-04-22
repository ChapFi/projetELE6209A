import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from Landmark import Landmark, Landmarks
import pandas as pd
from LazyData import LazyData
from tqdm import tqdm
from extraction import extract_trees
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.stats import chi2


Trees = np.array([])
currentTime = 0

def parse_sensor_management(filepath):
    """
    Parses a sensor management text file with the specified structure.

    Args:
        filepath (str): The path to the sensor management text file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the file.
              Returns None if the file cannot be opened or parsed.
    """
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if not lines:
            return []  # Return empty list if file is empty

        # Skip the header line
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) == 3:
                    try:
                        time = float(parts[0])
                        sensor = int(parts[1])
                        index = int(parts[2])
                        data.append({'time': time, 'sensor': sensor, 'index': index})
                    except ValueError:
                        print(f"Warning: Invalid data in line: {line}")
                        # Optionally, you can choose to raise an exception or skip the line
                else:
                    print(f"Warning: Incorrect number of columns in line: {line}")
                    #Optionally, you can choose to raise an exception or skip the line.

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def parse_laser_data(filepath):
    """
    Parses a laser sensor data file with the specified structure.

    Args:
        filepath (str): The path to the laser sensor data file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the file.
              Returns None if the file cannot be opened or parsed.
    """
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if not lines:
            return []  # Return empty list if file is empty

        # Skip the header line
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split(',')
                if len(parts) >= 362:  # Time + 361 laser points
                    try:
                        time_laser = float(parts[0])
                        laser_values = [float(val) for val in parts[1:]]  # Convert laser values to floats
                        data.append({'time_laser': time_laser, 'laser_values': laser_values})
                    except ValueError:
                        print(f"Warning: Invalid data in line: {line}")
                        # Optionally, you can choose to raise an exception or skip the line.
                else:
                    print(f"Warning: Incorrect number of columns in line: {line}")
                    #Optionally, you can choose to raise an exception or skip the line.

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def parse_odometry_data(filepath):
    """
    Parses an odometry sensor data file with the specified structure.

    Args:
        filepath (str): The path to the odometry sensor data file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the file.
              Returns None if the file cannot be opened or parsed.
    """
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if not lines:
            return []  # Return empty list if file is empty

        # Skip the header line
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) == 3:
                    try:
                        time_vs = float(parts[0])
                        velocity = float(parts[1])
                        steering = float(parts[2])
                        data.append({'time_vs': time_vs, 'velocity': velocity, 'steering': steering})
                    except ValueError:
                        print(f"Warning: Invalid data in line: {line}")
                        # Optionally, you can choose to raise an exception or skip the line.
                else:
                    print(f"Warning: Incorrect number of columns in line: {line}")
                    #Optionally, you can choose to raise an exception or skip the line.

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

L = 2.83  # Wheelbase
H = 0.76
b = 0.5
nbTrees = 0

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

def g(mesureT, estimatet1,dt, N):
    vt = mesureT["velocity"]
    wt = mesureT["steering"]
    theta = estimatet1[2]
    Dmu = np.array([vt * (-np.sin(theta) + np.sin(theta + wt * dt)) / wt,
                    vt * (np.cos(theta) - np.cos(theta + wt * dt)) / wt,
                    wt*dt])
    Fx = np.zeros((N, 3))
    Fx[0:3,0:3] = np.eye(3)
    newmu = estimatet1 + Fx @ Dmu
    return newmu



def predict_covariance(sigma, vt, wt, theta, dt, R_robot):
    """
    Blockwise prediction of the full (3+2n)x(3+2n) covariance matrix.

    sigma    : current full covariance (shape [3+2n,3+2n])
    vt, wt   : control inputs (velocity, steering rate)
    theta    : current robot heading
    dt       : time step
    R_robot  : 3×3 process‐noise for the robot state
    """
    N_full = sigma.shape[0]
    n_land = (N_full - 3) // 2

    # 1) Build the 3×3 motion‐Jacobian G_r (∂g/∂x)
    G_r = np.array([
        [1, 0, vt * (-np.cos(theta) + np.cos(theta + wt*dt)) / wt],
        [0, 1, vt * (-np.sin(theta) + np.sin(theta + wt*dt)) / wt],
        [0, 0, 1]
    ])

    # 2) Slice out robot–robot and robot–landmark blocks
    Σ_rr = sigma[0:3,     0:3    ]   # 3×3
    Σ_rL = sigma[0:3,     3:     ]   # 3×(2n)
    # Σ_LL = sigma[3:, 3:]           # (2n)×(2n), remains unchanged

    # 3) Propagate only those slices
    Σ_rr_new = G_r @ Σ_rr @ G_r.T + R_robot
    Σ_rL_new = G_r @ Σ_rL

    # 4) Write back in place
    sigma[0:3,     0:3    ] = Σ_rr_new
    sigma[0:3,     3:     ] = Σ_rL_new
    sigma[3:     , 0:3    ] = Σ_rL_new.T
    # (sigma[3:,3:] stays as is)

    return sigma

def normalize_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def compute_H_j(mu, j):
    """
    Build the 2 x (3+2N) Jacobian for the j-th landmark observation.
    mu:  state vector [x,y,theta, x1,y1, ..., xN,yN]
    j:   landmark index in [0..N-1]
    """
    x, y, theta = mu[0], mu[1], mu[2]
    xj = mu[3 + 2 * j]
    yj = mu[3 + 2 * j + 1]

    dx = xj - x
    dy = yj - y
    q = dx * dx + dy * dy
    r = np.sqrt(q)

    # Precompute partials
    dr_dx = -dx / r
    dr_dy = -dy / r
    dr_dtheta = 0.0
    dr_dxj = dx / r
    dr_dyj = dy / r

    dphi_dx = dy / q
    dphi_dy = -dx / q
    dphi_dtheta = -1.0
    dphi_dxj = -dy / q
    dphi_dyj = dx / q

    # Fill H: 2 rows, 3+2N cols
    D = len(mu)
    H = np.zeros((2, D))

    # Robot‐pose block
    H[0, 0:3] = [dr_dx, dr_dy, dr_dtheta]
    H[1, 0:3] = [dphi_dx, dphi_dy, dphi_dtheta]

    # Landmark j block
    base = 3 + 2 * j
    H[0, base: base + 2] = [dr_dxj, dr_dyj]
    H[1, base: base + 2] = [dphi_dxj, dphi_dyj]

    return H


from scipy.spatial import cKDTree

def data_association(mu, Sigma, measurements, R, n_active,
                                merge_radius=1.0,
                                gate_prob=0.95, ambiguity_ratio=1.2):
    # 1) Pre‑extract pose and small covariance blocks
    x, y, theta = mu[0], mu[1], mu[2]
    Sigma_rr = Sigma[0:3,   0:3]
    Sigma_rl = Sigma[0:3,   3:]
    Sigma_lr = Sigma_rl.T

    # 2) Build k-d tree on current landmark centers
    if n_active > 0:
        lm_pos = mu[3:3+2*n_active].reshape(n_active,2)
        tree  = cKDTree(lm_pos)
    else:
        lm_pos = np.zeros((0,2))
        tree   = None

    gamma = chi2.ppf(gate_prob, df=2)
    associations = []

    for (r_meas, phi_meas) in measurements:
        # approximate global point
        px = x + r_meas * np.cos(theta + phi_meas)
        py = y + r_meas * np.sin(theta + phi_meas)

        # ——  A) Euclidean pre‑merge  ——
        if tree is not None:
            idxs = tree.query_ball_point([px,py], merge_radius)
            if idxs:
                # pick the nearest by Eucl distance
                dists = [(px-lm_pos[j,0])**2 + (py-lm_pos[j,1])**2
                         for j in idxs]
                j_best = idxs[int(np.argmin(dists))]
                associations.append(j_best)
                continue

        # ——  B) full Mahalanobis gating  ——
        best_d2   = float("inf")
        second_d2 = float("inf")
        best_j    = None

        for j in range(n_active):
            xj = mu[3+2*j]
            yj = mu[3+2*j+1]
            dx, dy = xj - x, yj - y
            q = dx*dx + dy*dy
            if q < 1e-6:
                continue      # skip degenerate
            r_hat = np.sqrt(q)
            # range partials
            dr_dx      = -dx/r_hat
            dr_dy      = -dy/r_hat
            dr_dtheta  = 0.0
            dr_dxj     = +dx/r_hat
            dr_dyj     = +dy/r_hat
            a = np.array([dr_dx, dr_dy, dr_dtheta])   # 1×3
            b = np.array([dr_dxj, dr_dyj])            # 1×2

            # bearing partials
            dphi_dx     =  dy/q
            dphi_dy     = -dx/q
            dphi_dtheta = -1.0
            dphi_dxj    = -dy/q
            dphi_dyj    = +dx/q
            c = np.array([dphi_dx, dphi_dy, dphi_dtheta])
            d = np.array([dphi_dxj, dphi_dyj])

            # 3) extract the tiny covariance subblocks for landmark j
            idx3 = 3 + 2*j
            Sigma_rl_j = Sigma_rl[:,   2*j:2*j+2]          # shape (3,2)
            Sigma_lr_j = Sigma_lr[2*j:2*j+2, :]             # shape (2,3)
            Sigma_ll_j = Sigma[idx3:idx3+2, idx3:idx3+2]    # shape (2,2)

            # 4) form the 2×2 S directly
            # S11 = a Σ_rr aᵀ + 2 a Σ_rl_j bᵀ + b Σ_ll_j bᵀ
            S11 = a @ (Sigma_rr @ a) \
                 + 2*(a @ (Sigma_rl_j @ b)) \
                 + b @ (Sigma_ll_j @ b)
            # S22 = c Σ_rr cᵀ + 2 c Σ_rl_j dᵀ + d Σ_ll_j dᵀ
            S22 = c @ (Sigma_rr @ c) \
                 + 2*(c @ (Sigma_rl_j @ d)) \
                 + d @ (Sigma_ll_j @ d)
            # S12 = a Σ_rr cᵀ + a Σ_rl_j dᵀ + b Σ_lr_j cᵀ + b Σ_ll_j dᵀ
            S12 = a @ (Sigma_rr @ c) \
                 +   a @ (Sigma_rl_j @ d) \
                 +   b @ (Sigma_lr_j @ c) \
                 +   b @ (Sigma_ll_j @ d)

            S = np.array([[S11, S12],
                          [S12, S22]]) + R

            # 5) innovation
            z_hat = np.array([r_hat,
                              normalize_angle(np.arctan2(dy,dx) - theta)])
            y_err = np.array([r_meas, phi_meas]) - z_hat
            y_err[1] = normalize_angle(y_err[1])

            # 6) Mahalanobis distance without inverting S explicitly
            sol = np.linalg.solve(S, y_err)
            d2 = float(y_err.dot(sol))
            # track best and second‑best d2
            if d2 < best_d2:
                second_d2, best_d2 = best_d2, d2
                best_j = j
            elif d2 < second_d2:
                second_d2 = d2

        # gating logic
        if best_j is None or best_d2 > gamma:
            # truly new
            associations.append(None)
        elif second_d2 < gamma and second_d2/best_d2 < ambiguity_ratio:
            associations.append('discard')
        else:
            associations.append(best_j)

    return associations


def update_landmark_subblock(state, sigma, j, meas, Qt):
    """
    Perform the EKF correction for a single 2‑DOF landmark observation,
    touching only the 5×5 sub‐matrix for robot + that landmark.

    state:  (3+2n,) full state vector
    sigma:  (3+2n,3+2n) full covariance
    j:      landmark index (0‑based)
    meas:   (r,θ) measured range & bearing to landmark j
    Qt:     2×2 measurement noise
    """
    Nfull = state.shape[0]
    # 1) build the sub‑state indices: robot [0,1,2] + landmark j [3+2j, 4+2j]
    iL = 3 + 2 * j
    inds = [0, 1, 2, iL, iL + 1]

    # 2) extract sub‑state & sub‑covariance
    x_sub = state[inds].copy()  # shape (5,)
    Σ_sub = sigma[np.ix_(inds, inds)].copy()  # shape (5,5)

    # 3) compute expected measurement & Jacobian in this 5‑D space
    dx, dy = x_sub[3] - x_sub[0], x_sub[4] - x_sub[1]
    q = dx * dx + dy * dy
    sqrtq = np.sqrt(q)
    z_hat = np.array([sqrtq,
                      np.arctan2(dy, dx) - x_sub[2]])
    # 2×5 low‑dim Jacobian
    H_low = (1 / q) * np.array([
        [-sqrtq * dx, -sqrtq * dy, 0, sqrtq * dx, sqrtq * dy],
        [dy, -dx, -q, -dy, dx]
    ])

    # 4) Kalman gain in the 5‑D subspace
    S = H_low @ Σ_sub @ H_low.T + Qt  # 2×2
    K_sub = Σ_sub @ H_low.T @ np.linalg.inv(S)  # 5×2

    # 5) state & sub‑covariance update
    err = np.array(meas) - z_hat
    x_sub += K_sub @ err  # (5,)
    Σ_sub = (np.eye(5) - K_sub @ H_low) @ Σ_sub  # (5,5)

    # 6) scatter back into full state & covariance
    state[inds] = x_sub
    sigma[np.ix_(inds, inds)] = Σ_sub

    # 7) off‑diagonal blocks between these 5 rows and the rest:
    #    Σ[inds, rest] = (I−K H_low) Σ_sub,rest
    #    Σ[rest, inds] = transpose of above
    rest = list(set(range(Nfull)) - set(inds))
    Σ_old = sigma[np.ix_(inds, rest)].copy()  # (5,M)
    Σ_new = (np.eye(5) - K_sub @ H_low) @ Σ_old
    sigma[np.ix_(inds, rest)] = Σ_new
    sigma[np.ix_(rest, inds)] = Σ_new.T
    return state, sigma


def extendedKalman(state, u, sigma, measure, dt):
    N = (len(state))
    global currentTime
    currentTime = currentTime + dt
    newState = g(u, state, dt, N)
    # Define your robot process noise once, e.g.
    R_x = np.diag([0.5 ** 2, 0.5 ** 2, (0.5 * np.pi / 180) ** 2])

    # … inside your EKF predict …
    newSigma = predict_covariance(sigma,
                               vt=u['velocity'],
                               wt=u['steering'],
                               theta=state[2],
                               dt=dt,
                               R_robot=R_x)

    # #Correction
    # sr2 = 1 #To DO trouver les valeurs
    # sp2 = 1
    # Qt = np.array([[sr2,0],[0,sp2]])
    # for z in measure:
    #
    #     j, assoc_lm = landmarks.add(z, state, sigma)
    #     r, theta = assoc_lm.getDistance()
    #
    #     if np.linalg.norm(state[2*j+3:2*j+5]) == 0:
    #         estState[2*j+3:2*j+5] = np.array(estState[0:2]) + np.array([r * np.cos(theta + state[2]), r * np.sin(theta + state[2])])
    #
    #     update_landmark_subblock(estState, estSigma, j, (r,theta), Qt=np.eye(2))
    #
    # newState = estState
    # newSigma = estSigma
    return newState, newSigma

def updateEKF(state, sigma, measure: list[Landmark]):
    # Only update state and covariance when getting laser measurement
    # 1) unpack all (r,theta) from your Landmark objects
    zs = []
    for lm in measure:
        # assuming lm.getDistance() returns (r,theta)
        zs.append(tuple(lm.getDistance()))

    # 2) gate & associate them
    #    R: your 2×2 measurement noise; Qt below is the process/measurement noise you pass to update
    n_active = len(landmarks.landmarks)

    # associate
    associations = data_association(
        state, sigma, zs,
        R=np.eye(2) * 0.01,
        n_active=n_active,
        gate_prob=0.8, ambiguity_ratio=1
    )

    # 3) loop & update
    for (z, assoc, lm) in zip(zs, associations, measure):
        if assoc == 'discard':
            continue

        # brand-new?
        if assoc is None:
            # initialize a new landmark (augment mu,Sigma) and get its index j
            landmarks.add(lm)
            j = len(landmarks.landmarks)-1
            r, theta = z
            state[2*j+3:2*j+5] = state[0:2] + np.array([r*np.cos(theta + state[2]), r*np.sin(theta+state[2])])

        else:
            j = assoc

        # now do your usual EKF update on subblock j
        state, sigma = update_landmark_subblock(
            state, sigma, j, z,
            Qt=np.eye(2) * 0.01  # or whatever your laser‐noise is
        )
    return state, sigma


def EKFSlam(rowOdom, rowLaser, sensorManager):

    #Initiale state
    nbLandmark = 500
    X = np.zeros(3+2*nbLandmark)
    state = X
    sigma = np.eye(3+2*nbLandmark)*1e6
    sigma[:3, :3] = np.zeros((3, 3))
    currentTime = 0
    R_x = np.diag([0.5 ** 2, 0.5 ** 2, (0.5 * np.pi / 180) ** 2])

    robot_hist = []
    # lm_centers_hist = []
    # sigma_hist = []

    for entry in tqdm(sensorManager):
        if entry['sensor'] == 2:
            # — Prediction step
            u = rowOdom[entry['index']]
            dt = u['time_vs'] - currentTime
            currentTime += dt

            state = g(u, state, dt, len(state))
            sigma = predict_covariance(sigma,
                                       vt=u['velocity'],
                                       wt=u['steering'],
                                       theta=state[2],
                                       dt=dt,
                                       R_robot=R_x)

            robot_hist.append(state[:3].copy())
            # lm_centers_hist.append([
            #     (lm.centerx, lm.centery) for lm in landmarks.landmarks
            # ])
            # sigma_hist.append(sigma.copy())
        elif entry['sensor'] == 3:
            laser = rowLaser[entry['index']]['laser_values']
            # replace 30+ lines of manual masking with:
            detections = extract_trees(np.array(laser), params=None)

            z = []
            x_r, y_r, theta = state[0], state[1], state[2]
            for (r, ang, diam) in detections:
                cx = x_r + r * np.cos(ang + theta)
                cy = y_r + r * np.sin(ang + theta)
                z.append(Landmark(diameter=diam,
                                  center=(cx, cy),
                                  r=r,
                                  theta=ang))

            # now do your EKF update as before:
            updateEKF(state, sigma, z)
            robot_hist.append(state[:3].copy())
            # lm_centers_hist.append([
            #     (lm.centerx, lm.centery) for lm in landmarks.landmarks
            # ])
            # sigma_hist.append(sigma.copy())

    # return state, np.array(robot_hist), lm_centers_hist, sigma_hist
    return state, np.array(robot_hist), sigma

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
        cx, cy = lm.centerx, lm.centery

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

    landmarks = Landmarks()
    dataManagement = parse_sensor_management("dataset/Sensors_manager.txt")
    laserData = LazyData('dataset/LASER_processed.txt', "laser")
    drsData = LazyData('dataset/DRS.txt', "drs")
    #displayRowData(drsData, laserData, dataManagement)

    # import cProfile, pstats
    
    # cProfile.run('EKFSlam(drsData, laserData, dataManagement)', 'prof')
    # p = pstats.Stats('prof')
    # p.sort_stats('tottime').print_stats(10)
    finalstate, allState, sigma = EKFSlam(drsData, laserData, dataManagement)
    plt.plot(allState[:,0], allState[:,1], label='EKF SLAM')
    fig, ax = plt.subplots(figsize=(8, 8))
    # 1) plot robot path
    ax.plot(allState[:, 0], allState[:, 1], '-k', label='EKF Path')

    # 2) overlay landmarks + 2‑sigma ellipses
    plot_landmarks_with_cov(
        landmarks, sigma, ax=ax,
        n_std=2,
        edgecolor='blue', facecolor='none', linewidth=1, alpha=0.7
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('SLAM Landmarks with 95% Covariance Ellipses')
    ax.legend()
    ax.axis('equal')

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

    plt.savefig('foo.png')


