import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    #######################################
    # TO DO ...

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    ######################################
    # TO DO ...
    
    return pano_im


def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''

    ######################################
    # TO DO ...


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    generatePanaroma(im1, im2)