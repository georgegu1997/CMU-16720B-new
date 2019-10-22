import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    # M here is already
    # 1+p1 p2 p3
    # p4 1+p5 p6
    # 0    0  1
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Get the interpolation
    th, tw = It.shape
    ih, iw = It1.shape
    interp_It = RectBivariateSpline(np.arange(th), np.arange(tw), It)
    interp_It1 = RectBivariateSpline(np.arange(ih), np.arange(iw), It1)

    # Create the gridmesh for the template image
    x = np.arange(0, tw)
    y = np.arange(0, th)
    xt, yt = np.meshgrid(x, y)
    xt = xt.reshape(-1)
    yt = yt.reshape(-1)
    # Get the original coordinates
    Xt = np.vstack([xt, yt, np.ones(xt.shape)])

    # Image gradients do not change during the update
    # Therefore we can sample the image gradient before the iteration
    y_grad = interp_It1(yt, xt, dx=1, grid=False)
    x_grad = interp_It1(yt, xt, dy=1, grid=False)

    # The matrix A_{inv} and the Hessian A_{inv}^T A_{inv} can also be computed
    # before the iteration
    A = np.vstack([
        x_grad*xt,
        x_grad*yt,
        x_grad,
        y_grad*xt,
        y_grad*yt,
        y_grad]).T

    # Also compute the It part of b
    b0 = interp_It(yt, xt, grid=False)

    while True:
        # Get the warpped coordinates
        WX = M.dot(Xt)
        Wx = WX[0,:]
        Wy = WX[1,:]

        # Get the valid pixels that still remain in the second image
        idx_x_valid = np.logical_and(Wx >= 0, Wx < iw)
        idx_y_valid = np.logical_and(Wy >= 0, Wy < ih)
        idx_valid = np.logical_and(idx_x_valid, idx_y_valid)

        # Construct b
        b = (- b0 + interp_It1(Wy, Wx, grid=False)).reshape(-1,1)

        # remove the index that samples our of boundary here
        A_valid = A[idx_valid]
        b_valid = b[idx_valid]

        # Solve the least square problem
        dp, _, _, _ = np.linalg.lstsq(A_valid, b_valid, rcond=None)

        old_M = M.copy()

        # Compute the new M
        dM = np.eye(3) + np.vstack([dp.reshape(2, 3), np.zeros((1, 3))])
        M = M.dot(np.linalg.inv(dM))

        change = np.linalg.norm(dp)
        if change < 0.05:
            break
    return M[:2, :]
