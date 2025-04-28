import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from ekf import EKF
from extraction import extract_trees
from tqdm import tqdm
from Landmark import Landmarks, Landmark


class SLAM:
    def __init__(self, gps_parser, odom_parser, laser_parser, sensor_parser, R_robot, Qt):
        self.sensor_seq = sensor_parser
        self.gps_data = gps_parser
        self.odom_data = odom_parser
        self.laser_data = laser_parser
        self.ekf = EKF(dim_state=3+2*1500, R_robot=R_robot, Qt=Qt)
        self.landmarks = Landmarks()

    def run(self, history=True):
        currentTime = 0.0
        if history:
            all_state = [self.ekf.state[:3].copy()]
        for entry in tqdm(self.sensor_seq):
            dt = entry['time'] - currentTime
            currentTime += dt

            if entry['sensor'] == 1:
                gps = self.gps_data[entry['index']]
                self.ekf.update_gps((gps['longitude'], gps['latitude']), 2.5)

            elif entry['sensor'] == 2:
                # — Prediction step
                u = self.odom_data[entry['index']]
                state = self.ekf.predict(u, dt)

            elif entry['sensor'] == 3:
                laser = self.laser_data[entry['index']]['laser_values']

                # replace 30+ lines of manual masking with:
                detections = extract_trees(np.array(laser), params=None)

                z = []
                x_r, y_r, theta = self.ekf.state[0], self.ekf.state[1], self.ekf.state[2]
                for (r, ang, diam) in detections:
                    cx = x_r + r * np.cos(ang + theta)
                    cy = y_r + r * np.sin(ang + theta)
                    z.append(Landmark(diameter=diam,
                                      center=(cx, cy),
                                      r=r,
                                      theta=ang))

                # now do your EKF update as before:
                self.update_landmarks(z)

            if history:
                all_state.append(self.ekf.state[:3].copy())
        if history:
            return self.ekf.state, self.ekf.sigma, np.array(all_state)
        return self.ekf.state, self.ekf.sigma

    def update_landmarks(self, measure: list[Landmark]):
        # Only update state and covariance when getting laser measurement
        # 1) unpack all (r,theta) from your Landmark objects
        zs = []
        for lm in measure:
            # assuming lm.getDistance() returns (r,theta)
            r, theta = lm.getDistance()
            diam = lm.diameter  # grab the cluster‐based trunk width
            zs.append((r, theta, diam))

        # 2) gate & associate them
        #    R: your 2×2 measurement noise; Qt below is the process/measurement noise you pass to update
        associations = self.compute_data_association(zs)

        # 3) loop & update
        for (z, assoc, lm) in zip(zs, associations, measure):
            if assoc == -2:
                continue
            r, theta, diam = z

            # brand-new?
            if assoc == -1:
                # initialize a new landmark (augment mu,Sigma) and get its index j
                self.landmarks.add(lm)
                j = len(self.landmarks) - 1

                self.ekf.add_landmark(r, theta, diam, j)
            else:
                j = assoc

            self.ekf.update_landmark(j, (r, theta))

    def compute_data_association(self, measurements):
        '''
        Computes measurement data association.

        Given a robot and map state and a set of (range,bearing) measurements,
        this function should compute a good data association, or a mapping from
        measurements to landmarks.

        Returns an array 'assoc' such that:
            assoc[i] == j if measurement i is determined to be an observation of landmark j,
            assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
            assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
        '''

        if len(self.landmarks) == 0:
            return [-1 for _ in measurements]

        n_lmark = len(self.landmarks)
        n_scans = len(measurements)

        alpha = chi2.ppf(0.99, 2)
        beta = chi2.ppf(0.999, 2)
        A = alpha * np.ones((n_scans, n_scans))

        M = self.ekf.compute_res(n_lmark, measurements)

        M_new = np.hstack((M, A))
        rows, cols = linear_sum_assignment(M_new)
        assoc = np.full(n_scans, -2, dtype=int)

        for i, c in zip(rows, cols):
            cost = M_new[i, c]
            if cost > beta:
                assoc[i] = -2  # too big → discard
            elif c < n_lmark:
                assoc[i] = c  # matched to existing j
            else:
                assoc[i] = -1  # matched to “new” column

        return assoc