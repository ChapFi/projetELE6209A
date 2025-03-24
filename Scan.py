import numpy as np
import sys
import matplotlib.pyplot as plt
from functools import partial

class Scan:

    def __init__(self):
        """
        Initializes an empty scan landmark list.
        """
        self.landmark = []  # [[x, y, node_id]]
        self.count_landmark = 0

    def add_landmark(self, graph, scan_data, current_node_id):
        to_add, link_nodes,  = 0,0#landmarkCompare
        if to_add == True :
            pose = [scan_data[0], scan_data[1], 0]
            node_id = graph.add_node(pose)
            self.landmark.append([ scan_data[0] , scan_data[1], node_id])
            graph.add_edge()
        else:
            pose = [scan_data[0], scan_data[1], 0]


    def update(self, new_data):
        """
        Update the estimate of landmark position
        """
        self.landmark = new_data

    def print_scan(self):
        """
        Prints the contents of the pose graph.
        """
        print("Landmarks:")
        for landmark in self.landmark:
            print(f"landmark at position x:{landmark[0]} y:{landmark[1]}")

def data_read(filepath):
    data = {"FLASER":[]}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            message_type = parts[0]
            if message_type == "FLASER":
                num_readings = int(parts[1])
                range_readings = [float(x) for x in parts[2:2 + num_readings]]
                data["FLASER"].append({
                    "num_readings": num_readings,
                    "range_readings": range_readings,
                    "x": float(parts[2 + num_readings]),
                    "y": float(parts[3 + num_readings]),
                    "theta": float(parts[4 + num_readings]),
                    "odom_x": float(parts[5 + num_readings]),
                    "odom_y": float(parts[6 + num_readings]),
                    "odom_theta": float(parts[7 + num_readings]),
                    "ipc_timestamp": float(parts[8 + num_readings]),
                    "ipc_hostname": parts[9 + num_readings],
                    "logger_timestamp": float(parts[10 + num_readings]),
                })
    return data

def proceed_data(data):
    results = []
    for scan in data["FLASER"]:
        angles = np.linspace(-np.pi/2,np.pi/2, 180)
        pX = scan["x"]
        pY = scan["y"]
        scanner = np.zeros((2,180))
        i = 0
        for range, angle in zip(angles, scan["range_readings"]):
            scanner[0,i] =pX + range*np.cos(angle)
            scanner[1,i] = pY + range*np.sin(angle)
            i+=1
        results.append(scanner)
    return results

### Perform ICP for scan matching###

def plot_data(data_1, data_2, label_1, label_2, markersize_1=8, markersize_2=8):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    if data_1 is not None:
        x_p, y_p = data_1
        ax.plot(x_p, y_p, color='#336699', markersize=markersize_1, marker='o', linestyle=":", label=label_1)
    if data_2 is not None:
        x_q, y_q = data_2
        ax.plot(x_q, y_q, color='orangered', markersize=markersize_2, marker='o', linestyle=":", label=label_2)
    ax.legend()
    return ax

def plot_values(values, label):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(values, label=label)
    ax.legend()
    ax.grid(True)
    plt.show()

def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []
    for i in range(p_size):
        p_point = P[:, i]
        min_dist = sys.maxsize
        chosen_idx = -1
        for j in range(q_size):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences

def dR(theta):
    return np.array([[-np.sin(theta), -np.cos(theta)],
                     [np.cos(theta),  -np.sin(theta)]])

def R(theta):
    theta = theta[0]
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def jacobian(x, p_point):
    theta = x[2]
    J = np.zeros((2, 3))
    J[0:2, 0:2] = np.identity(2)
    J[0:2, [2]] = dR(0).dot(p_point)
    return J

def error(x, p_point, q_point):
    rotation = R(x[2])
    translation = x[0:2]
    prediction = rotation.dot(p_point) + translation
    return prediction - q_point

def prepare_system(x, P, Q, correspondences, kernel=lambda distance: 1.0):
    H = np.zeros((3, 3))
    g = np.zeros((3, 1))
    chi = 0
    for i, j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        e = error(x, p_point, q_point)
        weight = kernel(e) # Please ignore this weight until you reach the end of the notebook.
        J = jacobian(x, p_point)
        H += weight * J.T.dot(J)
        g += weight * J.T.dot(e)
        chi += e.T * e
    return H, g, chi

def icp_least_squares(P, Q, iterations=30, kernel=lambda distance: 1.0):
    x = np.zeros((3, 1))
    chi_values = []
    x_values = [x.copy()]  # Initial value for transformation.
    P_values = [P.copy()]
    P_copy = P.copy()
    corresp_values = []
    for i in range(iterations):
        rot = R(x[2])
        t = x[0:2]
        correspondences = get_correspondence_indices(P_copy, Q)
        corresp_values.append(correspondences)
        H, g, chi = prepare_system(x, P, Q, correspondences, kernel)
        dx = np.linalg.lstsq(H, -g, rcond=None)[0]
        x += dx
        x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2])) # normalize angle
        chi_values.append(chi.item(0))
        x_values.append(x.copy())
        rot = R(x[2])
        t = x[0:2]
        P_copy = rot.dot(P.copy()) + t
        P_values.append(P_copy)
    corresp_values.append(corresp_values[-1])
    return P_values, chi_values, corresp_values, rot, t

def draw_correspondences(P, Q, correspondences, ax):
    label_added = False
    for i, j in correspondences:
        x = [P[0, i], Q[0, j]]
        y = [P[1, i], Q[1, j]]
        if not label_added:
            ax.plot(x, y, color='grey', label='correpondences')
            label_added = True
        else:
            ax.plot(x, y, color='grey')
    ax.legend()

def kernel(error):
    return min(1/np.linalg.norm(error), 1)

if __name__ == "__main__":
    data = data_read("dataset/test.txt")
    scan = proceed_data(data)
    
    

    correspondences = get_correspondence_indices(scan[1], scan[0])
    ax = plot_data(scan[1], scan[0],
               label_1='P centered',
               label_2='Q centered')
    draw_correspondences(scan[1], scan[0], correspondences, ax)


    P_values, chi_values, corresp_values, R_found, t_found = icp_least_squares(scan[1], scan[0], kernel=partial(kernel))
    plot_values(chi_values, label="chi^2")
    print(R_found)
    print(t_found)
    Scan_corrected = R_found.dot(scan[1]) + t_found
    plot_data(scan[0], Scan_corrected, "P: corrected data", "Q: true data")
    plot_data( Scan_corrected, scan[1], "P: moved data", "Q: true data")
    plt.show()