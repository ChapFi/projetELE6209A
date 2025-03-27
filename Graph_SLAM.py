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
    relative_pose = np.multiply(T1_inv, T2)
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
            _, current_node_id = addODOM(graph, measurement, previous_node_id)
            previous_node_id = current_node_id
    return True


def Graph_opti(initX, constraint, nbNode):
    tol = 10^-6
    xold = initX
    xnew = np.multiply(xold, 2)
    maxIter = 100
    nbIter = 0
    while np.linalg.norm(xnew-xold) > tol and nbIter < maxIter:
        N = nbNode
        b = np.zeros((3*N,1))
        H = np.zeros((3*N,3*N))
        for from_node_id, to_node_id, relative_pose, W in constraint:
            i = from_node_id
            j = to_node_id

            Xi = xold[3*i:3*i+3] #graph.get_node_pose(i)
            Xj = xold[3*j:3*j+3] #graph.get_node_pose(j)
            tij = np.array([[relative_pose[0,0]],[relative_pose[1,0]]])
            thetaij = relative_pose[2,0]

            Ri = np.array([[np.cos(Xi[2,0]), -np.sin(Xi[2,0])],
                          [np.sin(Xi[2,0]),np.cos(Xi[2,0])]])
            
            dRi = np.array([[-np.sin(Xi[2,0]), -np.cos(Xi[2,0])],
                           [np.cos(Xi[2,0]),-np.sin(Xi[2,0])]])
            
            Rij = np.array([[np.cos(thetaij), -np.sin(thetaij)]
                           ,[np.sin(thetaij),np.cos(thetaij)]])
            
            rijri = np.matmul(np.transpose(Rij),np.transpose(Ri))

            dri = np.matmul(np.matmul(np.transpose(Rij),np.transpose(dRi)), np.subtract(Xj[0:2],Xi[0:2]))

            A = np.array([[-rijri[0,0], -rijri[0,1] ,dri[0,0]], 
                          [-rijri[1,0], -rijri[1,1], dri[1,0]], 
                          [0, 0, -1]])
            B = np.array([[rijri[0,0], rijri[0,1], 0],
                        [rijri[1,0], rijri[1,1], 0],
                        [0, 0, 1]])
            
            H[3*i:3*i+3, 3*i:3*i+3] = np.add(H[3*i:3*i+3, 3*i:3*i+3], np.matmul(np.matmul(np.transpose(A),W),A))
            H[3*i:3*i+3, 3*j:3*j+3] = np.add(H[3*i:3*i+3, 3*j:3*j+3], np.matmul(np.matmul(np.transpose(A),W),B))
            H[3*j:3*j+3, 3*i:3*i+3] = np.add(H[3*j:3*j+3, 3*i:3*i+3], np.matmul(np.matmul(np.transpose(B),W),A))
            H[3*j:3*j+3, 3*j:3*j+3] = np.add(H[3*j:3*j+3, 3*j:3*j+3], np.matmul(np.matmul(np.transpose(B),W),B))

            t = np.matmul( np.transpose(Rij), np.subtract(np.matmul(np.transpose(Ri),np.subtract(Xj[0:2],Xi[0:2])),tij))
            theta =  normalize(Xj[2]-Xi[2]- thetaij)
            e = np.array([[t[0,0]],[t[1,0]],[theta[0]]])

            b[3*i:(3*i)+3] = np.add(b[3*i:(3*i)+3], np.matmul(np.matmul(np.transpose(A),W),e))
            b[3*j:(3*j)+3] = np.add(b[3*j:(3*j)+3], np.matmul(np.matmul(np.transpose(B),W),e))

        H[0:3, 0:3] = np.add(H[0:3, 0:3],np.eye(3))
        #print(H)
        deltaX = spsolve(H, -b)
        deltaX = np.reshape(deltaX, (3*N, 1))
        xold = xnew
        xnew = np.add(xnew,deltaX)
        nbIter += 1
    xopt = xnew
    Hopt = H
    Hopt[0:3, 0:3] -= np.eye(3)
    return xopt , Hopt

def normalize(angle):
	return np.arctan2(np.sin(angle),np.cos(angle))

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

    pose_array = []
    edge_array = []

    n = 3  # SE(3)

    with open("input_INTEL_g2o.g2o", mode="r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts[0] == "VERTEX_SE2":
                id_ = int(parts[1])
                values = list(map(float, parts[2:]))
                pose_array.append([id_, np.atleast_2d(values).T])
            elif parts[0] == "EDGE_SE2":
                id1 = int(parts[1])
                id2 = int(parts[2])
                transform = list(map(float, parts[3:6]))
                information_matrix = list(map(float, parts[6:]))
                triu0 = np.triu_indices(3, 0)
                tril1 = np.tril_indices(3, -1)

                mat = np.zeros((3, 3), dtype=np.float64)
                mat[triu0] = information_matrix
                mat[tril1] = mat.T[tril1]
                edge_array.append([id1, id2, np.atleast_2d(transform).T, mat])

    print(edge_array[0][2])
    graph = PosGraph.PoseGraph()



    for pose in pose_array:
        graph.add_node(pose[1])

    for edge in edge_array:
        graph.add_edge(edge[0], edge[1], edge[2], edge[3])

    nbNodes = graph.get_number_of_nodes()
    xstart = np.empty((3 * nbNodes, 1))
    i = 0
    for index, (id, node) in enumerate(graph.nodes.items()):
        xstart[i] = node[0]
        xstart[i + 1] = node[1]
        xstart[i + 2] = node[2]
        i += 3
    C = graph.edges
    x, H = Graph_opti(xstart, C, nbNodes)
    x = np.reshape(x, (nbNodes, 3))
    x = np.multiply(x, 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x[:,0], x[:,1], 'b-')
    plt.show()
    plt.close()
