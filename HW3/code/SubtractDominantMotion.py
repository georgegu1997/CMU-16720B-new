import numpy as np

from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
# from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1
	# Output:
	#	mask: [nxm]
    # put your implementation here

    mask = np.ones(image1.shape, dtype=bool)

    # First get the affine transformation matrix that warp image1 to image2
    M = LucasKanadeAffine(image1, image2)
    # M = InverseCompositionAffine(image1, image2)

    # convert M to homo form and get its inverse
    M_homo = np.vstack([M, np.array([0,0,1])])
    M_inv = np.linalg.inv(M_homo)

    # Warp the image towards image2
    # Note that the transformation matrix given by LucasKanadeAffine is based on (y, x)
    # But the affine_transform is based on (x, y)
    warped_image1 = affine_transform(image1.T, M_inv).T

    # Don't account for pixels that move out of image
    valid_mask = np.ones_like(image1)
    warped_valid_mask = affine_transform(valid_mask.T, M_inv).T

    # Debugging
    # mask = np.absolute(warped_image1 - image2)

    mask = np.absolute(warped_image1 - image2) > 0.1
    mask = np.logical_and(mask, warped_valid_mask)

    return mask
