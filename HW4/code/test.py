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

if __name__ == '__main__':
    main()
