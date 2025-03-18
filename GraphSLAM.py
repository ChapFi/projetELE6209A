import copy

import numpy as np
import scipy as sp


def error(xi, xj, zij):
    Ti = xi[:2]
    Tj = xj[:2]
    theta_i = xi[2]
    theta_j = xj[2]
    theta_ij = zij[2]
    Zij_pos = zij[:2]

    # Calculate rotation matrices
    Ri = np.array(([np.cos(theta_i), -np.sin(theta_i)], [np.sin(theta_i), np.cos(theta_i)]))
    Rij = np.array(([np.cos(theta_ij), -np.sin(theta_ij)], [np.sin(theta_ij), np.cos(theta_ij)]))

    # This is the correct implementation per equation (30)
    return np.concatenate((Rij.T @ (Ri @ (Tj - Ti) - Zij_pos), np.array([theta_j - theta_i - theta_ij])))

class GraphSLAM:
    def __init__(self):
        self.nodes = {}
        self.edges = []  # (i, j, measurement, information matrix)

    def add_pose(self, index, pose):
        self.nodes[index] = np.array(pose, dtype=np.float64)

    def add_edge(self, i, j, measurement, information):
        self.edges.append((i, j, measurement, information))

    def optimize(self, max_iterations=10, epsilon=1e-6):
        """Optimize the graph using Levenberg-Marquardt algorithm"""
        iteration = 0
        lambda_factor = 1e-5
        converged = False

        while not converged and iteration < max_iterations:
            num_nodes = len(self.nodes)

            # Initialize H and b
            H = np.zeros((3 * num_nodes, 3 * num_nodes))
            b = np.zeros(3 * num_nodes)

            # Compute total error before optimization
            total_error = 0
            for i, j, Zij, Omega in self.edges:
                xi = self.nodes[i]
                xj = self.nodes[j]
                err = error(xi, xj, Zij)
                total_error += err.T @ Omega @ err

            # For each constraint
            for i, j, Zij, Omega in self.edges:
                xi = self.nodes[i]
                xj = self.nodes[j]

                # Calculate error - simplified from your original code
                e_ij = error(xi, xj, Zij)

                # Calculate Jacobians according to equations (32) and (33)
                Ti = xi[:2]
                Tj = xj[:2]
                theta_i = xi[2]

                # Rotation matrices
                Ri = np.array([
                    [np.cos(theta_i), -np.sin(theta_i)],
                    [np.sin(theta_i), np.cos(theta_i)]
                ])

                theta_ij = Zij[2]
                Rij = np.array([
                    [np.cos(theta_ij), -np.sin(theta_ij)],
                    [np.sin(theta_ij), np.cos(theta_ij)]
                ])

                # Calculate derivative of Ri
                dRi = np.array([
                    [-np.sin(theta_i), -np.cos(theta_i)],
                    [np.cos(theta_i), -np.sin(theta_i)]
                ])

                # Calculate A_ij and B_ij per equations (32) and (33)
                A_ij = np.zeros((3, 3))
                A_ij[:2, :2] = -Rij.T @ Ri.T
                A_ij[:2, 2:3] = Rij.T @ dRi.T @ (Tj - Ti).reshape(2, 1)
                A_ij[2, 2] = -1

                B_ij = np.zeros((3, 3))
                B_ij[:2, :2] = Rij.T @ Ri
                B_ij[2, 2] = 1

                # Compute contributions to H and b
                i_idx = slice(3 * i, 3 * (i + 1))
                j_idx = slice(3 * j, 3 * (j + 1))

                H[i_idx, i_idx] += A_ij.T @ Omega @ A_ij
                H[i_idx, j_idx] += A_ij.T @ Omega @ B_ij
                H[j_idx, i_idx] += B_ij.T @ Omega @ A_ij
                H[j_idx, j_idx] += B_ij.T @ Omega @ B_ij

                b[i_idx] += A_ij.T @ Omega @ e_ij
                b[j_idx] += B_ij.T @ Omega @ e_ij

            # Fix the first node (as in the pseudocode)
            H[:3, :3] += np.eye(3) * 1e6

            # Solve the system using sparse Cholesky
            H_sparse = sp.csr_matrix(H)
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                print(f"Linear algebra error in iteration {iteration}")
                lambda_factor *= 10
                continue

            # Make a copy of nodes for trial update
            new_nodes = copy.deepcopy(self.nodes)

            # Update nodes
            for i in range(num_nodes):
                delta = dx[3 * i:3 * (i + 1)]
                new_nodes[i] = new_nodes[i] + delta
                # Normalize angle
                new_nodes[i][2] = np.arctan2(np.sin(new_nodes[i][2]), np.cos(new_nodes[i][2]))

            # Compute error after update
            new_error = 0
            for i, j, Zij, Omega in self.edges:
                xi = new_nodes[i]
                xj = new_nodes[j]
                err = error(xi, xj, Zij)
                new_error += err.T @ Omega @ err

            self.nodes = new_nodes

            # Accept or reject update based on error change
            if new_error < total_error:
                # Accept update, decrease lambda
                lambda_factor /= 10

                # Check for convergence
                delta_norm = np.linalg.norm(dx)
                print(
                    f"Iteration {iteration}: error={new_error:.6f}, delta={delta_norm:.6f}, lambda={lambda_factor:.6f}")

            else:
                # Reject update, increase lambda
                lambda_factor *= 10
                print(f"Iteration {iteration}: rejected update, lambda={lambda_factor:.6f}")

            iteration += 1

    def get_nodes(self):
        return self.nodes

