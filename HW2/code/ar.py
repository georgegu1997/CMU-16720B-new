import numpy as np
import cv2
import os
from planarH import computeH


def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...

    return R, t


def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    # TO DO ...

    return X


if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')

    #############################
    # TO DO ...
    # perform required operations and plot sphere
    