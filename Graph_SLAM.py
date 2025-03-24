import matplotlib.pyplot as plt
import numpy as np
import PosGraph
from scipy.sparse.linalg import spsolve

def getScan(filepath, numberofScan):

    data = {
        "PARAM": [],
        "SYNC": [],
        "ODOM": [],
        "FLASER": [],
    }
    i = 0
    with open(filepath, "r") as f:
        for line in f:
            if i <= numberofScan:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue

                parts = line.split()
                message_type = parts[0]

                if message_type == "ODOM":
                    data["ODOM"].append({
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "theta": float(parts[3]),
                        "tv": float(parts[4]),
                        "rv": float(parts[5]),
                        "accel": float(parts[6]),
                        "ipc_timestamp": float(parts[7]),
                        "ipc_hostname": parts[8],
                        "logger_timestamp": float(parts[9]),
                    })
                elif message_type == "FLASER":
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
                    i+=1
    return data

def create_2d_pose_matrix(x, y, theta):
    """Creates a 2D homogeneous transformation matrix."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, x],[s, c, y],[0, 0, 1]])

def calculate_relative_pose_2d(pose1, pose2):
    """Calculates the relative pose between two 2D poses."""
    T1 = create_2d_pose_matrix(pose1[0,0], pose1[1,0], pose1[2,0])
    T2 = create_2d_pose_matrix(pose2[0,0], pose2[1,0], pose2[2,0])
    T1_inv = np.linalg.inv(T1)
    relative_pose = np.dot(T1_inv, T2)
    return relative_pose

def addODOM(graph, measure, from_node):
    pose = np.array([[measure["x"]], [measure["y"]], [measure["theta"]]])
    to_node = graph.add_node(pose)
    prev_pose = graph.get_node_pose(from_node)
    relative_pose = calculate_relative_pose_2d(prev_pose, pose) #TO DO s'assurer que c'est la bonne composition
    information_matrix = np.diag([1.0, 1.0, 0.5]) #TO DO quel modèle
    graph.add_edge(from_node, to_node,relative_pose, information_matrix)
    return True, to_node

def add_measurements(graph, odometry_data, from_node):
    """
    Adds odometry measurements to the pose graph as nodes and edges.

    Args:
        graph (PoseGraph): The pose graph object.
        odometry_data (list): A list of odometry measurements (dictionaries).
    """

    if not odometry_data:
        return False # nothing to add

    previous_node_id = from_node

    for measurement in odometry_data:
        lastpos = graph.get_node_pose(previous_node_id)
        ex = measurement["x"] - lastpos[0]
        ey = measurement["y"] - lastpos[1]
        etheta = measurement["theta"] - lastpos[2]
        if ex >= 0.5 or ey >= 0.5 or etheta >= 0.5:
            print("new add")
            _, current_node_id = addODOM(graph, measurement, previous_node_id)
            previous_node_id = current_node_id
    return True


def Graph_opti(initX, constraint, nbNode):
    tol = 10^-6
    xold = initX
    xnew = 2*xold
    maxIter = 100
    nbIter = 0
    while np.norm(xnew-xold) > tol and nbIter < maxIter:
        N = nbNode #TO DO checker les tailles
        b = np.zeros((3*N,1))
        H = np.zeros((3*N,3*N))
        for from_node_id, to_node_id, relative_pose, W in constraint:
            i = from_node_id
            j = to_node_id
            Xi = graph.get_node_pose(i)
            Xj = graph.get_node_pose(j)
            tij = np.array([[relative_pose[0]],[relative_pose[1]]])
            thetaij = relative_pose[2]
            Ri = np.array( [[np.cos(Xi[2]), -np.sin(Xi[2])],[np.sin(Xi[2]),np.cos(Xi[2])]])
            dRi = np.array( [[-np.sin(Xi[2]), -np.cos(Xi[2])],[np.cos(Xi[2]),-np.sin(Xi[2])]])
            Rij = np.array( [ [np.cos(Xj[2]), -np.sin(Xj[2])],[np.sin(Xj[2]),np.cos(Xj[2])]])
            rijri = np.transpose(Rij)*np.transpose(Ri)
            dri = np.transpose(Rij)*dRi*(Xj[0:1]-Xi[0:1])
            A = np.array([[-rijri[0,0], -rijri[0,1] ,dri[0]], 
                          [-rijri[1,0], -rijri[1,1], dri[1]], 
                          [0, 0, -1]])
            B = np.array([[rijri[0,0], rijri[0,1], 0],
                        [rijri[1,0], rijri[1,1], 0],
                        [0, 0, -1]])
            H[i:i+3, i:i+3] += np.transpose(A)*W*A
            H[i:i+3, j:j+3] += np.transpose(A)*W*B
            H[j:j+3, i:i+3] += np.transpose(B)*W*A
            H[j:j+3, j:j+3] += np.transpose(B)*W*B

            t = np.transpose(Rij)*(np.transpose(Ri)*(Xj[0:1]-Xi[0:1] - tij))
            theta = Xj[2]-Xi[2]- thetaij
            e = np.array([[t[0]],[t[1]],[theta]])
            b[i:i+3] += np.transpose(A)*W*e
            b[j:j+3] += np.transpose(B)*W*e
        H[0:2, 0:2] += np.eye(2)
        deltaX = spsolve(H, -b)
        xold = xnew
        xnew = xnew + deltaX
        nbIter += 1
    xopt = xnew
    Hopt = H
    Hopt[0:2, 0:2] -= np.eye(2)
    return xopt , Hopt

def initGraph(startPos):
    graph = PosGraph.PoseGraph()
    node_id = graph.add_node(startPos)
    return graph, node_id

def displayMAP(data):

    N = len(data["ODOM"])
    pos = np.zeros((N, 2))
    i = 0
    for point in data["ODOM"]:
        pos[i, 0] = point["x"]
        pos[i, 1] = point["y"]
        i += 1
    
    M = 180*len(data["FLASER"])
    posLidar = np.zeros((M,2))
    i = 0
    for measure in data["FLASER"]:
        angles = np.linspace(0,np.pi, 180)
        pX = measure["x"]
        pY = measure["y"]
        for range, angle in zip(angles, measure["range_readings"]):
            posLidar[i, 0] = pX + range*np.cos(angle)
            posLidar[i, 1] = pY + range*np.sin(angle)
            i += 1
    
    fig, ax = plt.subplots()
    plt.plot(pos[:,0], pos[:,1], '+')
    #plt.plot(posLidar[:,0], posLidar[:,1], '.')
    plt.show()
    return pos

if __name__ == "__main__": 
    data = getScan("dataset/aces_dataset.txt", 7000)
    x0 = np.array([[0],[0],[0]])
    graph, from_node = initGraph(x0)
    _ = add_measurements(graph, data["ODOM"][1:], from_node)
    #graph.print_graph()
    nbNodes = graph.get_number_of_nodes()
    xstart = np.array
    for _, node in enumerate(graph.nodes):
        np.append(xstart, node)
    C = graph.edges
    x, H = Graph_opti(xstart, C, nbNodes)