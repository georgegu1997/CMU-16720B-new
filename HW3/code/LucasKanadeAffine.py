import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

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
    Xt = np.vstack([xt, yt])

    while True:
        # Get the warpped coordinates
        WX = M[:2, :2].dot(Xt) + M[:2, 2].reshape(-1,1)
        Wx = WX[0,:]
        Wy = WX[1,:]

        # Get the valid pixels that still remain in the second image
        idx_x_valid = np.logical_and(Wx >= 0, Wx < iw)
        idx_y_valid = np.logical_and(Wy >= 0, Wy < ih)
        idx_valid = np.logical_and(idx_x_valid, idx_y_valid)

        Wx_valid = Wx[idx_valid]
        Wy_valid = Wy[idx_valid]
        xt_valid = xt[idx_valid]
        yt_valid = yt[idx_valid]

        # Get the image gradient of the current image at the warpped coordinates
        y_grad = interp_It1(Wy_valid, Wx_valid, dx=1, grid=False)
        x_grad = interp_It1(Wy_valid, Wx_valid, dy=1, grid=False)

        # Image gradient times affine transformation gradient and construct A
        A = np.vstack([
            x_grad*xt_valid,
            x_grad*yt_valid,
            x_grad,
            y_grad*xt_valid,
            y_grad*yt_valid,
            y_grad]).T
        # Construct b
        b = (interp_It(yt_valid, xt_valid, grid=False) - interp_It1(Wy_valid, Wx_valid, grid=False)).reshape(-1,1)

        # Solve the least square problem
        dp, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        old_M = M.copy()
        M += dp.reshape(2, 3)

        change = np.linalg.norm(dp)
        if change < 0.05:
            break

    return M
