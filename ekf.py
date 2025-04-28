import numpy as np
from scipy.stats import chi2

from Landmark import Landmark


def normalize_angle(angle: float) -> float:
    """Wraps angle to [-pi, pi)."""
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def calculate_nis(innovation: np.ndarray, cov: np.ndarray) -> float:
    """Normalized Innovation Squared (NIS)."""
    return innovation.T @ np.linalg.inv(cov) @ innovation



class EKF:
    def __init__(self, dim_state: int, R_robot: np.ndarray, Qt: np.ndarray, L=2.83, a=3.78, b=0.5, H=0.76):
        self.state = np.zeros(dim_state)
        self.sigma = np.eye(dim_state) * 1e6
        self.R_robot = R_robot
        self.Qt = Qt
        self.current_time = 0.0
        self.L = L
        self.a = a
        self.b = b
        self.H = H
        self.sigma[:3, :3] = np.diag([.1, .1, 1])

    def predict(self, control: dict, dt: float):
        # Propagate state
        alpha = control['steering']
        vt = control['velocity'] / (1 - np.tan(alpha) * self.H / self.L)
        x, y, theta = self.state[:3]

        Dmu = np.array([
            dt*(vt*np.cos(theta) - vt/self.L*np.tan(alpha)*(self.a*np.sin(theta)+self.b*np.cos(theta))),
            dt*(vt*np.sin(theta) + vt/self.L*np.tan(alpha)*(self.a*np.sin(theta)-self.b*np.cos(theta))),
            dt*vt/self.L*np.tan(alpha)
        ])
        Fx = np.zeros((len(self.state), 3)); Fx[0:3, 0:3] = np.eye(3)
        self.state += Fx @ Dmu
        self.state[2] = normalize_angle(self.state[2])


        # 1) Build the 3×3 motion‐Jacobian G_r (∂g/∂x)
        G_r = np.array([
            [1, 0, - dt * (vt * np.sin(theta) + vt / self.L * np.tan(alpha) * (self.a * np.cos(theta) - self.b * np.sin(theta)))],
            [0, 1, dt * (vt * np.cos(theta) - vt / self.L * np.tan(alpha) * (self.a * np.sin(theta) + self.b * np.cos(theta)))],
            [0, 0, 1]
        ])

        # 2) Slice out robot–robot and robot–landmark blocks
        Σ_rr = self.sigma[0:3, 0:3]  # 3×3
        Σ_rL = self.sigma[0:3, 3:]  # 3×(2n)
        # Σ_LL = sigma[3:, 3:]           # (2n)×(2n), remains unchanged

        # 3) Propagate only those slices
        Σ_rr_new = G_r @ Σ_rr @ G_r.T + self.R_robot
        Σ_rL_new = G_r @ Σ_rL

        # 4) Write back in place
        self.sigma[0:3, 0:3] = Σ_rr_new
        self.sigma[0:3, 3:] = Σ_rL_new
        self.sigma[3:, 0:3] = Σ_rL_new.T
        # (sigma[3:,3:] stays as is)

        self.current_time += dt

    def update_gps(self, gps_xy: tuple[float,float], sigma_gps: float, gate_prob: float = 0.999):
        # GPS-only update on x,y
        Pxy = self.sigma[:2, :2]
        P0 = self.sigma[:, :2]
        res = np.array(gps_xy) - self.state[:2]
        R = np.eye(2) * sigma_gps**2
        S = Pxy + R
        v = res.T @ np.linalg.inv(S) @ res
        if v > chi2.ppf(gate_prob, df=2):
            return
        K = P0 @ np.linalg.inv(S)
        self.state += K @ res
        self.state[2] = normalize_angle(self.state[2])
        self.sigma -= K @ P0.T


    def update_landmark(self, j: int, meas: tuple[float,float]):
        # Single landmark EKF update on indices [0,1,2,3+2j,4+2j]
        Nfull = self.state.shape[0]
        z, H = self.predict_landmark_measurement(j)
        inds = [0,1,2,3+2*j,4+2*j]
        Σ_sub = self.sigma[np.ix_(inds, inds)]
        S = H @ Σ_sub @ H.T + self.Qt
        K = Σ_sub @ H.T @ np.linalg.inv(S)
        innovation = np.array(meas) - z
        self.state[inds] += K @ innovation
        self.state[2] = normalize_angle(self.state[2])
        Σ_new = (np.eye(len(inds)) - K @ H) @ Σ_sub
        self.sigma[np.ix_(inds, inds)] = Σ_new
        rest = list(set(range(Nfull)) - set(inds))
        Σ_old = self.sigma[np.ix_(inds, rest)].copy()  # (5,M)
        Σ_new = (np.eye(5) - K @ H) @ Σ_old
        self.sigma[np.ix_(inds, rest)] = Σ_new
        self.sigma[np.ix_(rest, inds)] = Σ_new.T

    def predict_landmark_measurement(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the measurement and Jacobian for the j-th landmark:
          z_hat (2,), H (2 × len(mu))
        """
        x, y, theta = self.state[0], self.state[1], self.state[2]
        xj = self.state[3 + 2 * j]
        yj = self.state[3 + 2 * j + 1]

        dx = xj - x
        dy = yj - y
        q = dx * dx + dy * dy
        r = np.sqrt(q)

        # Predicted measurement
        z_hat = np.array([r, normalize_angle(np.arctan2(dy, dx) - theta)])

        H = (1 / q) * np.array([
            [-r * dx, -r * dy, 0, r * dx, r * dy],
            [dy, -dx, -q, -dy, dx]
        ])

        return z_hat, H

    def compute_res(self, n_lmark: int, measurements):
        n_scans = len(measurements)
        Qt = np.array([[0.1 ** 2, 0], [0, (1.25 * np.pi / 180) ** 2]])
        M = np.full((n_scans, n_lmark), 1e8, dtype=float)

        for j in range(n_lmark):
            iL = 3 + 2 * j
            inds = [0, 1, 2, iL, iL + 1]

            Σ_sub = self.sigma[np.ix_(inds, inds)].copy()  # shape (5,5)
            z, H = self.predict_landmark_measurement(j)

            # 4) Kalman gain in the 5‑D subspace
            S = H @ Σ_sub @ H.T + Qt  # 2×2
            Sinv = np.linalg.inv(S)
            for i in range(n_scans):
                temp_z = measurements[i][:2]
                res = temp_z - z
                M[i, j] = res.T @ (Sinv @ res)
        return M

    def add_landmark(self, r, theta, diam, j):
        self.state[2 * j + 3:2 * j + 5] = self.state[0:2] + np.array(
            [(r + diam / 2) * np.cos(theta + self.state[2]), (r + diam / 2) * np.sin(theta + self.state[2])])