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
import data_Visualisation as viz
from scipy.stats import chi2
import time

chi = chi2.ppf(0.999, df=2)

def solve_cost_matrix_heuristic(M):
    n_msmts = M.shape[0]
    result = []

    ordering = np.argsort(M.min(axis=1))

    for msmt in ordering:
        match = np.argmin(M[msmt,:])
        M[:, match] = 1e8
        result.append((msmt, match))

    return result

Trees = np.array([])
currentTime = 0

def normalize_angle(a):
    while a >= np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return a

def calculate_nis(innovation, innovation_covariance):
  """Calculates the Normalized Innovation Squared (NIS)."""
  return innovation.T @ np.linalg.inv(innovation_covariance) @ innovation

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
a = 3.78
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
    alpha = mesureT["steering"]
    vt = mesureT["velocity"]/(1- np.tan(alpha)*H/L)
    x, y, theta = estimatet1[:3]
    Dmu = np.array([dt*(vt*np.cos(theta) - vt/L*np.tan(alpha)*(a*np.sin(theta)+b*np.cos(theta))),
                    dt*(vt*np.sin(theta) + vt/L*np.tan(alpha)*(a*np.sin(theta)-b*np.cos(theta))),
                    dt*vt/L*np.tan(alpha)])
    Fx = np.zeros((N, 3))
    Fx[0:3,0:3] = np.eye(3)
    newmu = estimatet1 + Fx @ Dmu
    newmu[2] = normalize_angle(newmu[2])
    return newmu

def predict_covariance(sigma, vt, alpha, theta, dt, R_robot):
    """
    Blockwise prediction of the full (3+2n)x(3+2n) covariance matrix.

    sigma    : current full covariance (shape [3+2n,3+2n])
    vt, wt   : control inputs (velocity, steering rate)
    theta    : current robot heading
    dt       : time step
    R_robot  : 3×3 process‐noise for the robot state
    """
    N_full = sigma.shape[0]

    # 1) Build the 3×3 motion‐Jacobian G_r (∂g/∂x)
    G_r = np.array([
        [1, 0, - dt*(vt*np.sin(theta) + vt/L*np.tan(alpha)*(a*np.cos(theta)-b*np.sin(theta)))],
        [0, 1, dt*(vt*np.cos(theta) - vt/L*np.tan(alpha)*(a*np.sin(theta)+b*np.cos(theta)))],
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

def compute_data_association(state, sigma, measurements):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if len(landmarks.landmarks) == 0:
        return [-1 for _ in measurements]

    n_lmark = len(landmarks.landmarks)
    n_scans = len(measurements)
    M = np.full((n_scans, n_lmark), 1e8, dtype=float)
    Qt = np.array([[0.1**2, 0], [0, (1.25*np.pi/180)**2]])

    alpha = chi2.ppf(0.99, 2)
    beta = chi2.ppf(0.999, 2)
    A = alpha * np.ones((n_scans, n_scans))

    for j in range(n_lmark):
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
                          normalize_angle(np.arctan2(dy, dx) - x_sub[2])])
        # 2×5 low‑dim Jacobian
        H = (1 / q) * np.array([
            [-sqrtq * dx, -sqrtq * dy, 0, sqrtq * dx, sqrtq * dy],
            [dy, -dx, -q, -dy, dx]
        ])

        # 4) Kalman gain in the 5‑D subspace
        S = H @ Σ_sub @ H.T + Qt  # 2×2
        Sinv = np.linalg.inv(S)
        for i in range(n_scans):
            temp_z = measurements[i][:2]
            res = temp_z - np.squeeze(z_hat)
            M[i, j] = res.T @ (Sinv @ res)
 
    M_new = np.hstack((M, A))
    rows, cols = linear_sum_assignment(M_new)
    assoc = np.full(n_scans, -2, dtype=int)

    for i, c in zip(rows, cols):
        cost = M_new[i, c]
        if cost > beta:
            assoc[i] = -2  # too big → discard
        elif c < n_lmark:
            assoc[i] = c  # matched to existing j
        else:
            assoc[i] = -1  # matched to “new” column
    
    return assoc

def merge_landmarks(landmarks, pos_threshold=0.5, diam_threshold=0.5):
    merged = []
    skip_indices = set()

    for i, lm1 in enumerate(landmarks.landmarks):
        if i in skip_indices:
            continue

        for j in range(i+1, len(landmarks.landmarks)):
            if j in skip_indices:
                continue

            lm2 = landmarks.landmarks[j]

            # Mahalanobis distance between landmark centers
            delta = lm2.center - lm1.center
            combined_cov = lm1.covariance + lm2.covariance
            mahal_dist = delta.T @ np.linalg.inv(combined_cov) @ delta

            # Diameter difference
            diam_diff = abs(lm1.diameter - lm2.diameter)

            # Merge conditions
            if mahal_dist < pos_threshold and diam_diff < diam_threshold:
                lm1.update(lm2.center, lm2.diameter, lm2.covariance)
                skip_indices.add(j)
                del lm2

        merged.append(lm1)

    landmarks.landmarks = merged


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
                      normalize_angle(np.arctan2(dy, dx) - x_sub[2])])
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
    x_sub[2] = normalize_angle(x_sub[2])
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
    v = calculate_nis(err, S)
    return state, sigma, v

def gps_update(state: np.ndarray,
               sigma: np.ndarray,
               gps_xy: tuple[float,float],
               sigma_gps: float,
               gate_prob: float = 0.999):
    """
    Fast EKF GPS update on (x,y) only.
    """
    # 1) split off the top‐left 2×2 block and the first two columns
    P00 = sigma[:2,:2]             # 2×2
    P0  = sigma[:,:2]            # n×2

    # 2) innovation and noise
    res = np.array(gps_xy) - state[:2 ]# 2x1
    R = np.eye(2) * sigma_gps**2 # 2×2
    
    # 3) innovation cov & gate
    S = P00 + R
    Sinv = np.linalg.inv(S)
    v = res.T @ (Sinv @ res)

    if v > chi2.ppf(gate_prob, df=2):
        return state, sigma, None  # outlier, skip
    # 4) Kalman gain K = P[:,0:2] @ S⁻¹
    K = P0 @ Sinv           # n×2

    # 5) state update & wrap angle
    state += np.matmul(K, res)
    state[2] = normalize_angle(state[2])

    # 6) covariance update P ← P − K·P0ᵀ
    sigma -= K @ P0.T
    
    return state, sigma, v

def extendedKalman(state, u, sigma, measure, dt):
    N = (len(state))
    global currentTime
    currentTime = currentTime + dt
    newState = g(u, state, dt, N)
    # Define your robot process noise once, e.g.
    R_x = np.diag([0.15 ** 2, 0.15 ** 2, (0.5*np.pi/ 180) ** 2])# np.diag([0.05 ** 2, 0.05 ** 2, (0.5 * np.pi / 180) ** 2])#

    # … inside your EKF predict …
    newSigma = predict_covariance(sigma,
                                  vt=u['velocity'],
                                  alpha=u['steering'],
                                  theta=state[2],
                                  dt=dt,
                                  R_robot=R_x)

    return newState, newSigma

def updateEKF(state, sigma, measure: list[Landmark]):
    # Only update state and covariance when getting laser measurement
    # 1) unpack all (r,theta) from your Landmark objects
    zs = []
    for lm in measure:
        # assuming lm.getDistance() returns (r,theta)
        r, theta = lm.getDistance()
        diam = lm.diameter  # grab the cluster‐based trunk width
        zs.append((r, theta, diam))

    # 2) gate & associate them
    #    R: your 2×2 measurement noise; Qt below is the process/measurement noise you pass to update
    associations = compute_data_association(state, sigma, zs)

    # 3) loop & update
    Nis = []
    for (z, assoc, lm) in zip(zs, associations, measure):
        if assoc == -2:
            continue
        r, theta, diam = z

        # brand-new?
        if assoc == -1:
            # initialize a new landmark (augment mu,Sigma) and get its index j
            landmarks.add(lm)
            j = len(landmarks.landmarks)-1
            state[2*j+3:2*j+5] = state[0:2] + np.array([(r+diam/2)*np.cos(theta + state[2]), (r+diam/2)*np.sin(theta+state[2])])
        else:
            j = assoc

        state, sigma, nis = update_landmark_subblock(
            state, sigma, j, (r, theta),
            Qt= np.array([[0.1**2, 0], [0, (1.25*np.pi/180)**2]]) # np.array([[1**2, 0], [0, (0.01)**2]])#
        )
        landmarks.landmarks[j].update(state[2*j+3:2*j+5], diam, sigma[3 + 2 * j:5 + 2 * j, 3 + 2 * j:5 + 2 * j])
        updated_covariance = sigma[3 + 2 * j:5 + 2 * j, 3 + 2 * j:5 + 2 * j]
        landmarks.landmarks[j].covariance = updated_covariance
        Nis.append(nis)
    return state, sigma, Nis

def EKFSlam(rowGPS, rowOdom, rowLaser, sensorManager):

    #Initiale state
    nbLandmark = 1500
    X = np.zeros(3+2*nbLandmark)
    X[0] = -67.6493   
    X[1] = -41.7142 
    X[2] = 35.5*np.pi/180
    state = X
    sigma = np.eye(3+2*nbLandmark)*1e6
    sigma[0:3, 0:3] = np.diag([.1, .1, 1])
    currentTime = 0
    R_x = np.diag([0.15** 2, 0.15** 2, (0.5*np.pi/ 180) ** 2])# np.diag([0.05 ** 2, 0.05 ** 2, (0.5 * np.pi / 180) ** 2])#

    robot_hist = []
    i = 0
    j = 0

    state_hist = {
        't': [0],
        'NIS':[],
        'x': X[0:3],
        'cov': np.diag(sigma[0:3, 0:3])
    }

    time_stat = {
        'step':[],
        'time':[]
    }

    cov_trees = {
        't':[0],
        '1':[0, 0],
        '2': [0, 0],
        '3': [0, 0],
        '4': [0, 0],
        '5': [0, 0],
    }

    for entry in tqdm(sensorManager):
        dt = entry['time'] - currentTime
        currentTime += dt
        if i % 1000 == 0:
            viz.visualize_matrix_sns(sigma, len(landmarks.landmarks), j)
            viz.visualize_trajectorie(state_hist['x'], j)
            j += 1
        i += 1

        if entry['sensor'] == 2:
            # — Prediction step
            u = rowOdom[entry['index']]
            start_time = time.time()
            state = g(u, state, dt, len(state))
            sigma = predict_covariance(sigma,
                                       vt=u['velocity'],
                                       alpha=u['steering'],
                                       theta=state[2],
                                       dt=dt,
                                       R_robot=R_x)
            end_time = time.time()
            robot_hist.append(state[:3].copy())
            time_stat['time'].append(currentTime)
            time_stat['step'].append(end_time-start_time)

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
            start_time = time.time()
            state, sigma, Nis = updateEKF(state, sigma, z)
            end_time = time.time()
            for v in Nis:
                state_hist['NIS'].append(v)
            robot_hist.append(state[:3].copy())
            time_stat['time'].append(currentTime)
            time_stat['step'].append(end_time-start_time)

        elif entry['sensor'] == 1:
            gps = rowGPS[entry['index']]
            start_time = time.time()
            state, sigma, nis = gps_update(state, sigma, (gps['longitude'], gps['latitude']), 2.5)
            end_time = time.time()
            if nis != None:
                state_hist['NIS'].append(nis)
            time_stat['time'].append(currentTime)
            time_stat['step'].append(end_time-start_time)

        
        state_hist['x'] = np.vstack((state_hist['x'], state[0:3]))
        state_hist['cov'] = np.vstack((state_hist['cov'],np.diag(sigma[0:3, 0:3])))
        state_hist['t'].append(currentTime)
        cov_trees['1'] = np.vstack((cov_trees['1'], (np.diag(sigma[50:52, 50:52]) if (sigma[50,50]+sigma[51,51]) < 100 else [0,0])))
        cov_trees['2'] = np.vstack((cov_trees['2'], (np.diag(sigma[150:152, 150:152]) if (sigma[150,150]+sigma[151,151]) < 100 else [0,0])))
        cov_trees['3'] = np.vstack((cov_trees['3'], (np.diag(sigma[200:202, 200:202]) if (sigma[200,200]+sigma[201,201]) < 100 else [0,0])))
        cov_trees['4'] = np.vstack((cov_trees['4'], (np.diag(sigma[300:302, 300:302]) if (sigma[300,300]+sigma[301,301]) < 100 else [0,0])))
        cov_trees['5'] = np.vstack((cov_trees['5'], (np.diag(sigma[500:502, 500:502]) if (sigma[500,500]+sigma[501,501]) < 100 else [0,0])))
        cov_trees['t'].append(currentTime)

    #merge_landmarks(landmarks, pos_threshold=1, diam_threshold=0.5)

    return state, np.array(robot_hist), sigma, state_hist, cov_trees, time_stat


if __name__ == "__main__":

    landmarks = Landmarks()
    dataManagement = parse_sensor_management("dataset/Sensors_manager.txt")
    laserData = LazyData('dataset/LASER_processed.txt', "laser")
    drsData = LazyData('dataset/DRS.txt', "drs")
    gpsData = LazyData('dataset/GPS2.txt', 'gps')
    #viz.displayRowData(drsData, laserData, dataManagement)

    # import cProfile, pstats
    #
    # cProfile.run('EKFSlam(gpsData, drsData, laserData, dataManagement)', 'prof')
    # p = pstats.Stats('prof')
    # p.sort_stats('tottime').print_stats(10)
    finalstate, allState, sigma, hist, treeHist, timeHist = EKFSlam(gpsData, drsData, laserData, dataManagement)

    np.save('images/t_history.npy',hist['t'])
    np.save('images/x_history.npy', hist['x'])
    np.save('images/P_history.npy', hist['cov'])
    np.save('images/NIS_history.npy', hist['NIS'])
    print("State history arrays saved to individual .npy files.")
    
    print("Number of saved landamrks : ", len(landmarks.landmarks))

    # Final Map with covariance ellipse
    viz.plot_Map(allState, landmarks=landmarks, sigma=sigma)
    # Save perfromances on position estimations
    viz.plot_Perf(hist)
    # Proceed NIS test
    viz.NIS_test(hist["NIS"])
    # Covariance of 5 trees
    viz.plot_TreeCov(treeHist)
    #Time performance for computation
    viz.plot_Time(timeHist)