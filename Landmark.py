import numpy as np

class Landmark:
    def __init__(self, diameter, center, r, theta, covariance=np.eye(2) * 1e3):
        self.diameter = diameter
        self.center = np.array(center)
        self.covariance = covariance  # 2x2 position uncertainty covariance
        self.count = 1
        self.theta = theta
        self.r = r

    def update(self, new_center, new_diameter, new_cov):
        """Fuse new landmark into this landmark."""
        K = self.covariance @ np.linalg.inv(self.covariance + new_cov)
        self.center = self.center + K @ (new_center - self.center)
        self.covariance = (np.eye(2) - K) @ self.covariance
        self.diameter = (self.diameter * self.count + new_diameter) / (self.count + 1)
        self.count += 1

    def getDistance(self):
        return self.r, self.theta



class Landmarks:
    def __init__(self):
        self.landmarks: list[Landmark] = []

    def add(self, meas: Landmark, index=-1):
        """
        Associate or initialize a landmark using nearest-neighbor gating.
        meas: new Landmark with r, theta filled
        state: full state vector [x, y, theta, ...]
        sigma: full covariance
        """
        if index == -1:
            self.landmarks.append(meas)
        else:
            self.landmarks[index] = meas

    def __iter__(self):
        return iter(self.landmarks)

    def __len__(self):
        return len(self.landmarks)