import numpy as np
import matplotlib.pyplot as plt
from submission import *
from helper import *
import sys

def main():
    print("Load the data")
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    data = np.load("../data/some_corresp.npz")
    pts1 = data['pts1'].astype(float)
    pts2 = data['pts2'].astype(float)
    M = max(*im1.shape[:2])

    if sys.argv[1] == "2.1":
        print("Test the eightpoint() algorithm")
        F = eightpoint(pts1, pts2, M)
        print("fundamental matrix:")
        print(F)
        np.savez("../results/q2_1.npz", F=F, M=M)
        displayEpipolarF(im1, im2, F)

    elif sys.argv[1] == "2.2":
        print("test the sevenpoint() algorithm")
        F8p = eightpoint(pts1, pts2, M)
        print(F8p)

        # Set up data for evaluating F
        N = pts1.shape[0]
        homo1 = np.hstack([pts1, np.ones((N,1))])
        homo2 = np.hstack([pts2, np.ones((N,1))])

        # Sample 7 correspondences randomly
        # To expedite the searching process, I select F closest to that given by
        # the eightpoint() algorithm.
        best_F = None
        best_error = 1e10
        best_ind = None
        best_pts = None
        np.random.seed(2020)
        for _ in range(200):
            sample_ind = np.random.choice(np.arange(N), 7)
            sample_pts1 = pts1[sample_ind]
            sample_pts2 = pts2[sample_ind]
            Farray = sevenpoint(sample_pts1, sample_pts2, M)
            for F in Farray:
                error = np.linalg.norm(F - F8p)
                if error < best_error:
                    best_F = F
                    best_error = error
                    best_ind = sample_ind
                    best_pts = (sample_pts1, sample_pts2)
        print("Best sample indices:", best_ind)
        print("Best F from sampling with error:", best_error)
        print(best_F)
        np.savez("../results/q2_2.npz", F=best_F, M=M, pts1=best_pts[0], pts2=best_pts[1])

        displayEpipolarF(im1, im2, best_F)

    elif sys.argv[1] == "3.1":
        print("Test essentialMatrix() function")
        data = np.load("../data/intrinsics.npz")
        K1, K2 = data["K1"], data["K2"]
        F = eightpoint(pts1, pts2, M)
        E = essentialMatrix(F, K1, K2)
        print(E)

    elif sys.argv[1] == "4.1":
        F = eightpoint(pts1, pts2, M)
        np.savez("../results/q4_1.npz", F=F)
        epipolarMatchGUI(im1, im2, F)

    elif sys.argv[1] == "5.1":
        print("Test the ransacF() algorithm")
        some_corresp_noisy = np.load("../data/some_corresp_noisy.npz")
        pts1 = some_corresp_noisy['pts1'].astype(float)
        pts2 = some_corresp_noisy['pts2'].astype(float)
        F_ransac, inliers = ransacF(pts1, pts2, M)
        print(F_ransac)
        print(inliers.shape)
        # displayEpipolarF(im1, im2, F_ransac)

    elif sys.argv[1] == "5.2":
        print("Test rodrigues() and invRodrigues()")
        from scipy.stats import special_ortho_group
        for i in range(10):
            # Generate a random R
            R = special_ortho_group.rvs(3)
            r = invRodrigues(R)
            R_recon = rodrigues(r)
            print(np.linalg.norm(R-R_recon))
        for i in range(10):
            # Generate a random r
            r = np.random.rand(3,1)
            R = rodrigues(r)
            r_recon = invRodrigues(R)
            print(np.linalg.norm(r-r_recon))

    elif sys.argv[1] == "5.3":
        print("Test the rodriguesResidual()")
        some_corresp_noisy = np.load("../data/some_corresp_noisy.npz")
        pts1 = some_corresp_noisy['pts1'].astype(float)
        pts2 = some_corresp_noisy['pts2'].astype(float)
        intrinsics = np.load('../data/intrinsics.npz')
        K1, K2 = intrinsics["K1"], intrinsics["K2"]
        # estimate fundamental matrix
        F, inliers = ransacF(pts1, pts2, M)

        # Get only inliers for the following steps
        pts1, pts2 = pts1[inliers], pts2[inliers]

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

        R2 = M2[:, :3]
        t2 = M2[:, 3]
        r2 = invRodrigues(R2)

        x = np.concatenate([P.reshape(-1), r2.reshape(-1), t2])
        residuals = rodriguesResidual(K1, M1, pts1, K2, pts2, x)
        # print((residuals**2).reshape((-1, 2)))
        print(residuals.shape)
        print(np.linalg.norm(residuals)**2)


if __name__ == '__main__':
    main()
