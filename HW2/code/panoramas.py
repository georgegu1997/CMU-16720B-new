import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def createMask(im):
    mask = np.zeros((im.shape[0], im.shape[1]))
    mask[0,:] = 1
    mask[-1,:] = 1
    mask[:,0] = 1
    mask[:,-1] = 1
    mask = distance_transform_edt(1-mask)
    mask = mask/mask.max()
    return mask

def blendImages(im1, im2, im1_mask, im2_mask):
    total_weight = im1_mask + im2_mask
    im1_weight = np.zeros_like(total_weight)
    im2_weight = np.zeros_like(total_weight)
    im1_weight[im1_mask != 0] = im1_mask[im1_mask != 0] / total_weight[im1_mask != 0]
    im2_weight[im2_mask != 0] = im2_mask[im2_mask != 0] / total_weight[im2_mask != 0]
    im1, im2 = im1.astype(float), im2.astype(float)
    blend = np.expand_dims(im1_weight, axis=-1) * im1 + np.expand_dims(im2_weight, axis=-1) * im2
    blend = blend.round().astype(np.uint8)
    return blend

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
    output_shape = (int(im2.shape[1] * 1.7), im2.shape[0])

    # Create the mask for blending
    im2_mask = createMask(im2)
    im1_mask = createMask(im1)

    # warp the im2 and the mask of im2
    warp_im = cv2.warpPerspective(im2, H2to1, output_shape)
    cv2.imwrite('../results/6_1_warp.jpg', warp_im)
    warp_mask = cv2.warpPerspective(im2_mask, H2to1, output_shape)

    # Construct the panorama image and its mask
    pano_im = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)
    pano_mask = np.zeros((output_shape[1], output_shape[0]), dtype=float)
    pano_im[:im1.shape[0], :im1.shape[1]] = im1
    pano_mask[:im1.shape[0], :im1.shape[1]] = im1_mask

    # Blend the two images
    pano_im = blendImages(warp_im, pano_im, warp_mask, pano_mask)

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

    # Create the mask for blending
    im1_mask = createMask(im1)
    im2_mask = createMask(im2)

    # Construct the extreme points
    extreme_1 = [
    [0,0,1],
    [0,im1.shape[0],1],
    [im1.shape[1],0,1],
    [im1.shape[1],im1.shape[0],1],
    ]
    extreme_2 = [
        [0,0,1],
        [0,im2.shape[0],1],
        [im2.shape[1],0,1],
        [im2.shape[1],im2.shape[0],1],
    ]
    extreme_1 = np.array(extreme_1).T
    extreme_2 = np.array(extreme_2).T

    # Warp the extreme points for im2
    warped_extreme = H2to1.dot(extreme_2)
    warped_extreme = (warped_extreme / warped_extreme[2, :]).round()
    all_extreme = np.hstack([extreme_1, warped_extreme])

    left_top = all_extreme.min(axis=1)
    right_bottom = all_extreme.max(axis=1)

    # width, height and shifts for the output image
    w = int(right_bottom[0] - left_top[0])
    h = int(right_bottom[1] - left_top[1])
    tx = int(max(-left_top[0], 0))
    ty = int(max(-left_top[1], 0))

    # Only translation is used here
    M = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)

    warp_im1 = cv2.warpPerspective(im1, M, (w, h))
    warp_im1_mask = cv2.warpPerspective(im1_mask, M, (w, h))
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (w, h))
    warp_im2_mask = cv2.warpPerspective(im2_mask, np.matmul(M, H2to1), (w, h))

    # Blend the two images
    pano_im = blendImages(warp_im1, warp_im2, warp_im1_mask, warp_im2_mask)

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
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im

def main():
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

    # 6.1
    pano_im = imageStitching(im1, im2, H2to1)
    cv2.imwrite('../results/6_1_pan.jpg', pano_im)
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
