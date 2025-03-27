import numpy as np
from GraphSLAM import GraphSLAM
import matplotlib.pyplot as plt


pose_array = []
edge_array = []

n = 3 # SE(3)

with open("input_INTEL_g2o.txt", mode="r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if parts[0] == "VERTEX_SE2":
            id_ = int(parts[1])
            values = list(map(float, parts[2:]))
            pose_array.append([id_, values])
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
            edge_array.append([id1, id2, transform, mat])

print("Pose Array:", pose_array)
print("Edge Array:", edge_array)

graph = GraphSLAM()

for pose in pose_array:
    graph.add_pose(pose[0], pose[1])

for edge in edge_array:
    graph.add_edge(edge[0], edge[1], edge[2], edge[3])

graph.optimize()

opt_x, opt_y = [], []
for key, T in graph.get_nodes().items():
    opt_x.append(float(T[0]))
    opt_y.append(float(T[1]))

plt.figure(figsize=(10, 5))
plt.plot(opt_x, opt_y, 'b-')
plt.show()