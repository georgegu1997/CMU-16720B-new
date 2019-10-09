import numpy as np
import cv2
import os
from planarH import computeH

import matplotlib.pyplot as plt

def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...

    # pre-multiplying the estimated homography by the inverse of the intrinsic matrix
    K_inv = np.linalg.inv(K)
    H_prime = K_inv.dot(H)
    # compute the SVD of the first two columns of H_prime
    U, L, VH = np.linalg.svd(H_prime[:,:2], full_matrices=True)

    # find the closest valid first two columns of a rotation matrix
    R12 = U.dot(np.array([[1,0], [0,1], [0,0]])).dot(VH)
    R3 = np.cross(R12[:, 0], R12[:, 1])
    R3 = np.expand_dims(R3, axis=0).T
    R = np.hstack([R12, R3])

    #we test the determinant of the resulting rotation matrix, and if it is -1, we multiply the last column by -1.
    if np.linalg.det(R) == -1:
        R[:,-1] = -1 * R[:,-1]

    lambda_prime = (H_prime[:, :2] / R[:, :2]).sum() / 6
    t = H_prime[:, 2:3] / lambda_prime

    return R, t


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    # TO DO ...

    if W.shape[0] == 3:
        W = np.vstack([W, np.ones((1, W.shape[1]))])
    X = K.dot(np.hstack([R, t])).dot(W)
    X = X / X[2:3, :]
    return X


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')

    #############################
    # TO DO ...
    # perform required operations and plot sphere

    W = np.array([
        [0.0, 18.2, 18.2, 0.0],
        [0.0, 0.0, 26.0, 26.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    X = np.array([
        [483, 1704, 2175, 67],
        [810, 781, 2217, 2286]
    ])

    K = np.array([
        [3043.72, 0.0, 1196.00],
        [0.0, 3043.72, 1604.00],
        [0.0, 0.0, 1.0]
    ])

    # Copmute the homography H
    H = computeH(X, W[:2, :])
    R, t = compute_extrinsics(K, H)

    # Load the image and the points coordinates of the sphere
    points = np.loadtxt("../data/sphere.txt")

    # Find the location of "O"
    Xo = np.array([[830], [1640], [1]]) # location in image
    Wo = np.linalg.inv(H).dot(Xo)
    Wo = Wo / Wo[2:3, :] # location on the plane

    # Get the bottom of the sphere
    min_z = points[2,:].min()

    # Shift the points towards "O" and shift up
    shift = np.vstack([Wo[:2, :], -min_z])
    points_shifted = points + shift
    projected_points = project_extrinsics(K, points_shifted, R, t)

    plt.imshow(im[:,:,::-1])
    plt.plot(projected_points[0,:], projected_points[1,:], 'yo')
    plt.show()
