from matplotlib.patches import Ellipse

from LazyData import LazyData
from slam import SLAM
import numpy as np
import matplotlib.pyplot as plt


def plot_landmarks_with_cov(landmarks, sigma, ax=None, n_std=2.0, **ellipse_kwargs):
    """
    Plot landmarks and their covariance ellipses.

    landmarks : your Landmarks() instance
    sigma     : full (3+2n)x(3+2n) covariance matrix
    ax        : optional matplotlib Axes
    n_std     : number of standard deviations for the ellipse (e.g. 2 = 95% confidence)
    ellipse_kwargs : passed to Ellipse (edgecolor, facecolor='none', alpha, etc.)
    """
    ax = ax or plt.gca()

    for j, lm in enumerate(landmarks.landmarks):
        # 1) extract mean
        cx, cy = lm.center

        # 2) extract 2x2 covariance for this landmark
        iL = 3 + 2*j
        cov = sigma[iL:iL+2, iL:iL+2]

        # 3) eigen‐decompose to get axis lengths & orientation
        vals, vecs = np.linalg.eigh(cov)
        # sort descending so vals[0] is largest
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]
        # angle of largest eigenvector
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        # width & height of ellipse = 2 * n_std * sqrt(eigenvalue)
        width, height = 2 * n_std * np.sqrt(vals)

        # 4) create & add the ellipse patch
        ell = Ellipse((cx, cy), width, height, angle=angle, **ellipse_kwargs)
        ax.add_patch(ell)

        # 5) plot center point
        ax.plot(cx, cy, 'ro', markersize=3)

    return ax

if __name__ == '__main__':
    R_robot = np.diag([0.15**2,0.15**2,(0.5*np.pi/180)**2])
    Qt = np.array([[0.1**2,0],[0,(1.25*np.pi/180)**2]])

    slam = SLAM(
        gps_parser=LazyData('dataset/GPS.txt', "gps"),
        odom_parser=LazyData('dataset/DRS.txt', "drs"),
        laser_parser=LazyData('dataset/LASER_processed.txt', "laser"),
        sensor_parser=LazyData('dataset/Sensors_manager.txt', "manager"),
        R_robot=R_robot, Qt=Qt
    )
    state, sigma, allState = slam.run(history=True)
    plt.plot(allState[:, 0], allState[:, 1], label='EKF SLAM')
    fig, ax = plt.subplots(figsize=(8, 8))
    # 1) plot robot path
    ax.plot(allState[:, 0], allState[:, 1], '-k', label='EKF Path')

    # 2) overlay landmarks + 2‑sigma ellipses
    plot_landmarks_with_cov(
        slam.landmarks, sigma, ax=ax,
        n_std=2,
        edgecolor='blue', facecolor='none', linewidth=1, alpha=0.7
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('SLAM Landmarks with 95% Covariance Ellipses')
    ax.legend()
    ax.axis('equal')
    print(len(slam.landmarks))

    plt.savefig('foo.png')

    fig, ax = plt.subplots()
    im = ax.imshow(sigma, cmap='viridis',  # pick any Matplotlib colormap
                   interpolation='none'  # no smoothing between cells
                   )

    fig.colorbar(im, ax=ax)

    plt.savefig('cov.png')