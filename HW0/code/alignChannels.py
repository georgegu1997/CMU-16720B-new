import numpy as np
from scipy.signal import correlate2d as corr
from scipy.signal import convolve2d as conv
from scipy.ndimage import shift

import matplotlib.pyplot as plt

MAX_SHIFT = 30

'''
Return the offset of c2 related to c1
CC = uv / (u_norm * v_norm)
'''
def alignTwoNCC(c1, c2):
    h, w = c1.shape

    # Normalize the two image to float
    c1 = c1.astype(float) / 255.0
    c2 = c2.astype(float) / 255.0

    # Firstly pad the c1
    pad_c1 = np.pad(c1, MAX_SHIFT, 'wrap')

    # Compute the correlation of U * V
    uv = corr(pad_c1, c2, 'valid')

    # Compute u^2 + v^2 for each offset. Only take the overlapping region into account
    # Get the 2 norm of overlapping resgion using corr trick
    mask = np.ones_like(c1, dtype=float)
    mask = np.pad(mask, MAX_SHIFT, 'constant')
    u1 = np.flip(corr(mask, c1, 'valid')) # flip on the both dimension
    v1 = corr(mask, c2, 'valid')
    SSD = uv / (u1*v1)

    # Get the location of the max response and thus the shift offset
    argmax = np.unravel_index(SSD.argmax(), SSD.shape)
    argmax = np.array(argmax) - MAX_SHIFT

    return argmax

'''
Return the offset of c2 related to c1
For SSD: (u-v)^2 = u^2 - 2uv + v^2
'''
def alignTwoSSD(c1, c2):
    h, w = c1.shape

    # Normalize the two image to float
    c1 = c1.astype(float) / 255.0
    c2 = c2.astype(float) / 255.0

    # Firstly pad the c1
    pad_c1 = np.pad(c1, MAX_SHIFT, 'wrap')

    # Compute the correlation of U * V
    uv = corr(pad_c1, c2, 'valid')

    # Compute u^2 + v^2 for each offset. Only take the overlapping region into account
    # Get the 2 norm of overlapping resgion using corr trick
    mask = np.ones_like(c1, dtype=float)
    mask = np.pad(mask, MAX_SHIFT, 'constant')
    u2 = np.flip(corr(mask, c1**2, 'valid')) # flip on the both dimension
    v2 = corr(mask, c2**2, 'valid')
    SSD = u2+v2-2*uv

    # Get the location of the max response and thus the shift offset
    argmax = np.unravel_index(SSD.argmin(), SSD.shape)
    argmax = np.array(argmax) - MAX_SHIFT

    return argmax

def alignChannels(red, green, blue, alignFunc):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    # Take the red channel as the base
    g2r_offset = alignFunc(red, green)
    b2r_offset = alignFunc(red, blue)

    print(g2r_offset)
    print(b2r_offset)

    shifted_g = shift(green, g2r_offset)
    shifted_b = shift(blue, b2r_offset)

    image = np.stack((red, shifted_g, shifted_b), axis=-1)

    return image
