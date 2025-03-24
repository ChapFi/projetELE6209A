import numpy as np

class PoseGraph:
    """
    A class to represent a pose graph for SLAM.

    Attributes:
        nodes (dict): A dictionary to store nodes (poses) with node IDs as keys.
        edges (list): A list to store edges (constraints) between nodes.
        node_counter (int): A counter to assign unique IDs to nodes.
    """

    def __init__(self):
        """
        Initializes an empty pose graph.
        """
        self.nodes = {}  # {node_id: pose (numpy array)}
        self.edges = []  # [(from_node_id, to_node_id, relative_pose, information_matrix)]
        self.node_counter = 0

    def add_node(self, pose):
        """
        Adds a new node (pose) to the graph.

        Args:
            pose (numpy.ndarray): The pose of the new node (e.g., [x, y, theta] or [x, y, z, qx, qy, qz, qw]).

        Returns:
            int: The ID of the newly added node.
        """
        node_id = self.node_counter
        self.nodes[node_id] = pose.copy()  # Store a copy of the pose
        self.node_counter += 1
        return node_id

    def add_edge(self, from_node_id, to_node_id, relative_pose, information_matrix):
        """
        Adds an edge (constraint) between two nodes.

        Args:
            from_node_id (int): The ID of the source node.
            to_node_id (int): The ID of the destination node.
            relative_pose (numpy.ndarray): The relative pose between the two nodes.
            information_matrix (numpy.ndarray): The information matrix representing the uncertainty of the constraint.
        """
        self.edges.append((from_node_id, to_node_id, relative_pose.copy(), information_matrix.copy()))

    def get_node_pose(self, node_id):
      """
      Returns the pose of the specified node.

      Args:
          node_id (int): The ID of the node.
      Returns:
          numpy.ndarray: The pose of the node.
      """
      return self.nodes.get(node_id)

    def get_edge(self, index):
        """
        Returns the edge at the specified index.

        Args:
            index (int): The index of the edge.

        Returns:
            tuple: The edge (from_node_id, to_node_id, relative_pose, information_matrix).
        """
        if 0 <= index < len(self.edges):
            return self.edges[index]
        else:
            return None

    def get_number_of_nodes(self):
      """
      Returns the number of nodes in the graph.
      Returns:
        int: number of nodes.
      """
      return len(self.nodes)

    def get_number_of_edges(self):
      """
      Returns the number of edges in the graph.
      Returns:
        int: number of edges.
      """
      return len(self.edges)

    def print_graph(self):
        """
        Prints the contents of the pose graph.
        """
        print("Nodes:")
        for node_id, pose in self.nodes.items():
            print(f"  Node {node_id}: {pose}")
        print("Edges:")
        for from_id, to_id, rel_pose, info_matrix in self.edges:
            print(f"  Edge ({from_id} -> {to_id}):")
            print(f"    Relative Pose: {rel_pose}")
            print(f"    Information Matrix:\n{info_matrix}")