'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
import matplotlib.pyplot as plt
from submission import epipolarCorrespondence, eightpoint, essentialMatrix, triangulate
from helper import camera2
from findM2 import test_M2_solution

def main():
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    intrinsics = np.load('../data/intrinsics.npz')
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    some_corresp = np.load("../data/some_corresp.npz")
    pts1 = some_corresp['pts1'].astype(float)
    pts2 = some_corresp['pts2'].astype(float)
    M = max(*im1.shape[:2])

    # Since templeCoords does not include correspondences
    # the fundamental matrix should given by some_corresp
    F = eightpoint(pts1, pts2, M)
    E = essentialMatrix(F, K1, K2)

    templeCoords = np.load("../data/templeCoords.npz")
    x1 = templeCoords['x1']
    y1 = templeCoords['y1']

    # Get the correspondences for each point in x1
    X1 = np.hstack([x1, y1])
    X2 = []
    for (x1, y1) in X1:
        (x2, y2) = epipolarCorrespondence(im1, im2, F, x1, y1)
        X2.append([x2, y2])
    X2 = np.vstack(X2)

    # triangulate the correspondences

    # M1 is normalized
    M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    C1 = K1.dot(M1)
    # get the rotation and translation from essential matrix
    M2s = camera2(E)

    # Solve 3D points using each possible transform and get the best one
    best_P = None
    best_err = 1e10
    best_reproj_err = None
    best_M2 = None
    best_C2 = None
    for i in range(M2s.shape[2]):
        M2 = M2s[:,:,i]
        C2 = K2.dot(M2)
        P, reproj_err = triangulate(C1, X1, C2, X2)
        # Define the err to be the maximum distance in the points
        # reasonable solution does not come from a smaller reprojection error
        err = P[:, 0].max() - P[:, 0].min() + P[:, 1].max() - P[:, 1].min() + P[:, 2].max() - P[:, 2].min()
        if err < best_err:
            best_P = P
            best_err = err
            best_reproj_err = reproj_err
            best_M2 = M2
            best_C2 = C2
    M2 = best_M2
    C2 = best_C2
    P = best_P
    print("best_reproj_err:",best_reproj_err)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_xlim([-10,10])
    # ax.set_ylim([-5,5])
    # ax.set_zlim([-10,10])

    plt.show()

if __name__ == '__main__':
    main()
