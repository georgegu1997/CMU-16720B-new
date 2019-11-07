"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import matplotlib.pyplot as plt
from helper import refineF, displayEpipolarF
import sys

def main():
    print("Load the data")
    im1 = plt.imread("../data/im1.png")
    im2 = plt.imread("../data/im2.png")
    data = np.load("../data/some_corresp.npz")
    pts1 = data['pts1'].astype(float)
    pts2 = data['pts2'].astype(float)

    if sys.argv[1] == "2.1":
        print("Test the eightpoint() algorithm")
        M = max(*im1.shape[:2])
        F = eightpoint(pts1, pts2, M)
        print("fundamental matrix:")
        print(F)
        np.savez("../results/q2_1.npz", F=F, M=M)
        displayEpipolarF(im1, im2, F)

    elif sys.argv[1] == "2.2":
        print("test the sevenpoint() algorithm")
        F8p = np.load("../results/q2_1.npz")['F']
        print(F8p)
        M = max(*im1.shape[:2])

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
        F = np.load("../results/q2_1.npz")['F']
        E = essentialMatrix(F, K1, K2)
        print(E)

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # Determine the normalization matrix and scale all the points
    N, _ = pts1.shape
    T = np.array([
        [1.0/M, 0, 0],
        [0, 1.0/M, 0],
        [0, 0, 1.0]
    ])
    norm1, norm2 = pts1/M, pts2/M
    homo1 = np.hstack([norm1, np.ones((N, 1))])
    homo2 = np.hstack([norm2, np.ones((N, 1))])

    # Calculate F
    # Construct A
    A = np.hstack([homo2[:,0:1]*homo1, homo2[:,1:2]*homo1, homo1])

    # SVD and take the solution of f
    u, s, vh = np.linalg.svd(A)
    f = vh.T[:, 8]
    F = f.reshape((3,3))

    # Ensure rank(F) = 2
    u, s, vh = np.linalg.svd(F)
    s[-1] = 0
    F = u.dot(np.diag(s).dot(vh))

    # Refine F using refineF()
    F = refineF(F, norm1, norm2)

    # Rescale F using the normalization matrix
    F = T.T.dot(F).dot(T)

    # print(np.linalg.norm(homo2.dot(F).dot(homo1.T)))
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, 7x2 Matrix
            pts2, 7x2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # Determine the normalization matrix and scale all the points
    N, _ = pts1.shape
    T = np.array([
        [1.0/M, 0, 0],
        [0, 1.0/M, 0],
        [0, 0, 1.0]
    ])
    norm1, norm2 = pts1/M, pts2/M
    homo1 = np.hstack([norm1, np.ones((N, 1))])
    homo2 = np.hstack([norm2, np.ones((N, 1))])

    # Construct A
    A = np.hstack([homo2[:,0:1]*homo1, homo2[:,1:2]*homo1, homo1])

    # SVD and get the two null space f1, f2 of A
    u, s, vh = np.linalg.svd(A)
    f1 = vh.T[:, -1]
    f2 = vh.T[:, -2]
    F1, F2 = f1.reshape((3,3)), f2.reshape((3,3))

    # Finding the coefficient of det(a*F1 + (1-a)*F2)
    # Reference: https://piazza.com/class/jzslw1sd69v31q?cid=258
    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    a0 = fun(0)
    a1 = 2*(fun(1)-fun(-1))/3-(fun(2)-fun(-2))/12
    a2 = 0.5*fun(1)+0.5*fun(-1)-fun(0)
    a3 = fun(1)-a0-a1-a2

    # solve for the roots of the above polynomial
    roots = np.roots([a3, a2, a1, a0])
    # print(roots)

    # Construct the Farray by the real roots of alpha
    Farray = []
    for a in roots:
        if np.isreal(a):
            F = a*F1 + (1-a)*F2

            # # Refine F using refineF()
            # F = refineF(F, norm1, norm2)

            # Rescale F using the normalization matrix
            F = T.T.dot(F).dot(T)

            Farray.append(F)

    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T.dot(F).dot(K1)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    N, _ = pts1.shape
    # Construct A matrix
    A = []
    X1 = np.hstack([pts1, np.zeros((N, 2))]).reshape((-1,1))
    X2 = np.hstack([np.zeros((N, 2)), pts2]).reshape((-1,1))
    X = X1 + X2

    C_left = np.vstack(([C1[2]]*2 + [C2[2]]*2) * N)
    C_right = np.vstack([C1[:2], C2[:2]] * N)
    A = X*C_left - C_right

    # Solve for the 3D coordinate for each point respectively
    P = []
    err = 0
    for i in range(N):
        Ai = A[i*4:(i+1)*4]

        U, S, Vh = np.linalg.svd(Ai)
        wi = Vh.T[:,-1]
        # normalize
        wi = wi / wi[-1]
        P.append(wi[:3])

        # Calculate the reprojection error
        x1_proj = C1.dot(wi)
        x1_proj = x1_proj / x1_proj[-1]
        x2_proj = C2.dot(wi)
        x2_proj = x2_proj / x2_proj[-1]
        x1, x2 = pts1[i], pts2[i]
        this_err = np.linalg.norm(x1-x1_proj[:2])**2 + np.linalg.norm(x2-x2_proj[:2])**2
        # print(this_err)
        err += this_err

    P = np.stack(P, axis=0)
    # print(P.shape)
    # print(err)
    return P, err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass

if __name__ == '__main__':
    main()
