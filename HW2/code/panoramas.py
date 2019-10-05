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
    # Let the output to be larger than the input
    output_shape = (im2.shape[1] * 2, im2.shape[0])
    warp_im = cv2.warpPerspective(im2, H2to1, output_shape)

    pano_im = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)
    pano_im[:im1.shape[0], :im1.shape[1]] = im1
    # Using elementwise maximum to blend
    pano_im = np.maximum(warp_im, pano_im)

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

def main():
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

    # 6.1
    pano_im = imageStitching(im1, im2, H2to1)
    cv2.imwrite('../results/6_1.jpg', pano_im)
    np.save('../results/q6_1.npy', H2to1)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6.2 No cliping
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/6_2_pan.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6.3 all together
    pano_im = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
