import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                equation
    '''

    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    N = p1.shape[1]

    # \lambda p1 = H2to1 p2. Therefore, p1 = x, p2 = u
    # Construct the A matrix
    x = np.hstack([p1.T, np.ones((N, 1))])
    u = np.hstack([p2.T, np.ones((N, 1))])
    rpart1 = - u * x[:, 0:1]
    rpart2 = - u * x[:, 1:2]
    A = np.hstack([u, np.zeros((N, 3)), rpart1, np.zeros((N, 3)), u, rpart2])
    A = A.reshape([2*N, -1])

    # compute SVD for A
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    # Get the last row of vh and reshape to get H
    H2to1 = vh[-1, :].reshape((3, 3))
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC

    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''

    ###########################
    # TO DO ...

    bestH = []
    best_inliers = []
    sample_space = np.arange(0, matches.shape[0])

    # \lambda p1 = H2to1 p2. Therefore, p1 = x, p2 = u
    for i in range(num_iter):
        # Random sample the matches
        idx = np.random.choice(sample_space, size=4, replace=False)
        sampled_matches = matches[idx]
        sampled_locs1 = locs1[sampled_matches[:,0], :2]
        sampled_locs2 = locs2[sampled_matches[:,1], :2]

        # Compute and apply the transformation matrix H
        H2to1 = computeH(sampled_locs1.T, sampled_locs2.T)
        u = locs2[matches[:,1]].T
        u[2, :] = 1
        x = locs1[matches[:,0]].T
        x[2, :] = 1
        Hu = H2to1.dot(u)
        # Normalize Hu because of the lambda
        normalized_Hu = Hu / Hu[2,:]
        error = ((normalized_Hu-x)[:2, :] ** 2).sum(axis=0)

        # Get the index of inliers
        inlier_idx = error < tol
        n_inliers = inlier_idx.sum()
        inliers = matches[inlier_idx]
        if n_inliers > len(best_inliers):
            best_inliers = inliers
            bestH = H2to1

    print("RANSAC found the best H with %d/%d inliers" % (len(best_inliers), len(matches)))
    # print("Best H is:")
    # print(bestH)

    return bestH

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
