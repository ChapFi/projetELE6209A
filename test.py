import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def make_symmetric(P):
    return 0.5 * (P + P.T)


def clamp_angle(theta):
    while theta >= np.pi:
        theta -= 2*np.pi

    while theta < -np.pi:
        theta += 2*np.pi

    return theta


def to_center(ve):
    return ve/(1-np.tan(alpha)*H/L)


# Simulation parameters
dt = 0.1  # Time step
T = 50  # Total time in seconds
L = 2.83  # Wheelbase
H = 0.76
b = 0.5
alpha = 3.78 * np.pi/180
GPS_file = "dataset/GPS.txt"
odom_file = "dataset/DRS.txt"

df = pd.read_csv(GPS_file, delimiter = "\t", header=None)
GPS_data = df.iloc[:, 1:].to_numpy()

df = pd.read_csv(odom_file, delimiter = "\t", header=None)
odom_data = df.iloc[:, 1:].to_numpy()
deltaT = np.diff(df.iloc[:, 0].to_numpy())

num_steps = odom_data.shape[0]
P = np.diag([.1, .1, 1])

# EKF Dead Reckoning using Odometry
ekf_states = np.zeros((num_steps, 3))  # EKF estimated states [x, y, theta]
Q = np.diag([0.1, 0.1, np.radians(0.1)]) ** 2  # Process noise covariance
for t in range(1, num_steps):
    v = odom_data[t, 0]
    vc = to_center(v)
    delta = odom_data[t, 1]
    theta = ekf_states[t - 1, 2]

    deltax = deltaT[t-1] * (vc * np.cos(theta))
    deltay = deltaT[t-1] * (vc * np.sin(theta))
    deltatheta = clamp_angle((vc / L) * np.tan(delta))

    u = np.array([deltax, deltay, deltatheta])

    dx = -deltaT[t-1] * (vc * np.sin(theta))
    dy = deltaT[t-1] * (vc * np.cos(theta))
    F = np.eye(3)

    G = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    R = np.zeros((3, 3))
    R[0, 0] = 0.05**2
    R[1, 1] = 0.05**2
    R[2, 2] = 0.5*np.pi/180**2

    new_cov = make_symmetric(G @ (P @ G.T) + F.T @ (R @ F))

    # Prediction Step
    ekf_states[t, 0] = ekf_states[t - 1, 0] + deltaT[t-1] * (vc * np.cos(theta))
    ekf_states[t, 1] = ekf_states[t - 1, 1] + deltaT[t-1] * (vc * np.sin(theta))
    ekf_states[t, 2] = ekf_states[t - 1, 2] + (vc / L) * np.tan(delta) * deltaT[t-1]

    # Update covariance (no correction step since it's dead reckoning)
    P = new_cov

# Plot trajectory and comparison
plt.figure(figsize=(8, 6))
plt.scatter(GPS_data[:, 0], GPS_data[:, 1], label='Truth', s=1)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.plot(ekf_states[:, 0], ekf_states[:, 1], label='EKF Dead Reckoning', linestyle='--')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.title('EKF Dead Reckoning vs GPS Ground Truth')
plt.grid()
plt.show()