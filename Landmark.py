import numpy as np

class Landmark:
    def __init__(self, diameter, center, r=0, theta=0):
        self.diameter = diameter
        self.centerx, self.centery = center
        self.r = r
        self.theta = theta

    def __eq__(self, other):
        if not isinstance(other, Landmark):
            return False
        return (abs(self.diameter - other.diameter) < 1
                and abs(self.centerx - other.centerx) < 1
                and abs(self.centery - other.centery) < 1)

    def update(self, r, theta):
        self.r = r
        self.theta = theta

    def getDistance(self):
        return self.r, self.theta


def wrap_to_pi(angle):
    """Normalize angle into [\u2013\u03c0, \u03c0]."""
    return (angle + np.pi) % (2*np.pi) - np.pi

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