import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


L = 2.83  # Wheelbase
H = 0.76
b = 0.5


def ekf_prediction_step(state, control_input, delta_t, Q):
    """
    More standard EKF prediction step for Ackermann vehicle model

    Parameters:
    - state: Current state [x, y, theta]
    - control_input: [velocity, steering_angle]
    - delta_t: Time step
    - Q: Process noise covariance
    - L: Wheelbase

    Returns:
    - Predicted state
    - Prediction Jacobian
    """
    v, alpha = control_input
    x, y, theta = state

    # Center velocity transformation
    vc = v / (1 - np.tan(alpha) * H / L)

    # Motion model
    dx = vc * np.cos(theta) * delta_t
    dy = vc * np.sin(theta) * delta_t
    dtheta = (vc / L) * np.tan(alpha) * delta_t

    # State prediction
    new_state = np.array([
        x + dx,
        y + dy,
        (theta + dtheta + np.pi) % (2 * np.pi) - np.pi  # Normalized angle
    ])

    # Jacobian of motion model w.r.t state
    F = np.array([
        [1, 0, -vc * np.sin(theta) * delta_t],
        [0, 1, vc * np.cos(theta) * delta_t],
        [0, 0, 1]
    ])

    # Jacobian of motion model w.r.t noise
    W = np.eye(3)

    # Prediction covariance
    prediction_cov = F @ Q @ F.T + W @ Q @ W.T

    return new_state, prediction_cov


# Example usage would look like:
# initial_state = np.array([0, 0, 0])
# control_input = np.array([2.0, 0.1])  # velocity, steering angle
# Q = np.diag([0.1, 0.1, 0.01])  # process noise covariance
# dt = 0.1
# new_state, new_cov = ekf_prediction_step(initial_state, control_input, dt, Q, L=2.83)

GPS_file = "dataset/GPS.txt"
odom_file = "dataset/DRS.txt"

df = pd.read_csv(GPS_file, delimiter = "\t", header=None)
GPS_data = df.iloc[:, 1:].to_numpy()

df = pd.read_csv(odom_file, delimiter = "\t", header=None)
odom_data = df.iloc[:, 1:].to_numpy()
deltaT = np.diff(df.iloc[:, 0].to_numpy())

num_steps = odom_data.shape[0]

# EKF Dead Reckoning using Odometry
ekf_states = np.zeros((num_steps, 3))  # EKF estimated states [x, y, theta]
Q = np.diag([0.1, 0.1, np.radians(0.1)]) ** 2  # Process noise covariance

for t in range(1, num_steps):
    state = ekf_states[t-1, :]
    u = [odom_data[t, 0], odom_data[t, 1]]
    delta_t = deltaT[t - 1]
    ekf_states[t, :], Q = ekf_prediction_step(state, u, delta_t, Q)


print(Q)
# Plot trajectory and comparison
plt.figure(figsize=(8, 6))
plt.scatter(GPS_data[:, 1], GPS_data[:, 0], label='Truth (GPS)', s=1)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.plot(ekf_states[:, 0], ekf_states[:, 1], label='EKF Dead Reckoning', linestyle='--', color='r')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.title('EKF Dead Reckoning vs GPS Ground Truth')
plt.grid()
plt.show()

