import numpy as np
import matplotlib.pyplot as plt
from submission import eightpoint, essentialMatrix, triangulate
from helper import camera2

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

def test_M2_solution(pts1, pts2, intrinsics, M):
    '''
    Estimate all possible M2 and return the correct M2 and 3D points P
    :param pred_pts1:
    :param pred_pts2:
    :param intrinsics:
    :param M: a scalar parameter computed as max (imwidth, imheight)
    :return: M2, the extrinsics of camera 2
             C2, the 3x4 camera matrix
             P, 3D points after triangulation (Nx3)
    '''
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    # estimate fundamental matrix using eightpoint or sevenpoint algorithm
    F = eightpoint(pts1, pts2, M)

    # get essential matrix using fundamental and intrinsics
    E = essentialMatrix(F, K1, K2)

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
        P, reproj_err = triangulate(C1, pts1, C2, pts2)
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

    return M2, C2, P


if __name__ == '__main__':
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    M = max(*im1.shape[:2])
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    intrinsics = np.load('../data/intrinsics.npz')

    M2, C2, P = test_M2_solution(pts1, pts2, intrinsics, M)

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='o')
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # # ax.set_xlim([-10,10])
    # # ax.set_ylim([-5,5])
    # # ax.set_zlim([-10,10])
    #
    # plt.show()

    np.savez('q3_3', M2=M2, C2=C2, P=P)
