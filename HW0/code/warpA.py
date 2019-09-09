import numpy as np

def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_shape.
    Uses nearest neighbor interpolation."""

    # Get the mesh grid
    yy, xx = np.meshgrid(np.arange(output_shape[1]), np.arange(output_shape[0])) # x coordinate first
    grid = np.stack((xx, yy), axis=-1)
    grid = grid.reshape((-1, 2)).T
    P_dst = np.vstack((grid, np.ones((1, grid.shape[1]))))

    # Compute the P_src from P_dst
    # This is actually (y, x, 1)^T = A (y, x, 1)^T
    A_inv = np.linalg.inv(A)
    P_src = A_inv.dot(P_dst)
    P_src = P_src.round().astype(int)[:2, :]

    # Get the valid index
    valid_ind = np.logical_and(
        np.logical_and(P_src[0] >= 0, P_src[0] < im.shape[0]),
        np.logical_and(P_src[1] >= 0, P_src[1] < im.shape[1])
    )
    valid_P_src = P_src[:, valid_ind]

    # Index the input image to get the output warped image
    warped = np.zeros(output_shape).flatten()
    warped[valid_ind] = im[valid_P_src[0], valid_P_src[1]]
    warped = warped.reshape(output_shape)

    return warped
