import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1,
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''

    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here

    DoG_levels = levels[1:]
    DoG_pyramid = gaussian_pyramid[:,:,1:] - gaussian_pyramid[:,:,:-1]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None

    ##################
    # TO DO ...
    # Compute principal curvature here

    # Compute the Hessian matrix using second-order Sobel filters
    Dxx = cv2.Sobel(DoG_pyramid,cv2.CV_64F,2,0,ksize=3)
    Dxy = cv2.Sobel(DoG_pyramid,cv2.CV_64F,1,1,ksize=3)
    Dyy = cv2.Sobel(DoG_pyramid,cv2.CV_64F,0,2,ksize=3)

    # Compute the Principal Curvature using Hessian
    Tr = Dxx + Dyy
    Det = Dxx * Dyy - Dxy ** 2 + 1e-7
    principal_curvature = Tr**2 / Det

    # Take the absolute value and convert back to uint8
    principal_curvature = cv2.convertScaleAbs(principal_curvature)

    # The above line is equivalent to the following three lines
    # principal_curvature = np.absolute(principal_curvature)
    # principal_curvature = np.clip(principal_curvature, 0, 255)
    # principal_curvature = np.uint8(principal_curvature)

    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None

    ##############
    #  TO DO ...
    # Compute locsDoG here
    contrast_thresh = np.absolute(DoG_pyramid) > th_contrast
    curvature_thresh = np.absolute(principal_curvature) < th_r

    # Pad the DoG_pyramid with 0 on each border each axis
    padded = np.pad(DoG_pyramid, pad_width=1, mode="constant", constant_values=0)

    # construct an array to store the local extrema comparison results
    compare_max = np.zeros([10, *DoG_pyramid.shape])
    compare_min = np.zeros([10, *DoG_pyramid.shape])
    for c, a, p in [(compare_max, DoG_pyramid, padded), (compare_min, -DoG_pyramid, -padded)]:
        c[0] = a > p[2:, 1:-1, 1:-1] # (x, y, c) > (x+1, y, c)
        c[1] = a > p[2:, 2:, 1:-1] # (x, y, c) > (x+1, y+1, c)
        c[2] = a > p[1:-1, 2:, 1:-1] # (x, y, c) > (x, y+1, c)
        c[3] = a > p[:-2, 2:, 1:-1] # (x, y, c) > (x-1, y+1, c)
        c[4] = a > p[:-2, 1:-1, 1:-1] # (x, y, c) > (x-1, y, c)
        c[5] = a > p[:-2, :-2, 1:-1] # (x, y, c) > (x-1, y-1, c)
        c[6] = a > p[1:-1, :-2, 1:-1] # (x, y, c) > (x, y-1, c)
        c[7] = a > p[2:, :-2, 1:-1] # (x, y, c) > (x+1, y-1, c)
        c[8] = a > p[1:-1, 1:-1, 2:] # (x, y, c) > (x, y, c+1)
        c[9] = a > p[1:-1, 1:-1, :-2] # (x, y, c) > (x, y, c-1)

    # Logical AND and OR the local minima and local maxima
    compare = np.logical_or(np.all(compare_max, axis=0), np.all(compare_min, axis=0))

    # AND all the conditions
    local_extrema = np.logical_and(np.logical_and(compare, contrast_thresh), curvature_thresh)

    # Convert the extrema map to indexes
    locsDoG = np.array(np.nonzero(local_extrema))
    locsDoG = np.transpose(locsDoG, (1, 0))
    locsDoG[:,2] = np.array(DoG_levels)[locsDoG[:,2]]

    # x coordinate corresponds to columns and y coordinate corresponds to rows
    locsDoG[:, [0,1]] = locsDoG[:, [1, 0]]

    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''

    ##########################
    # TO DO ....
    # compupte gauss_pyramid, locsDoG here
    gauss_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    return locsDoG, gauss_pyramid

def drawPoints(im, locs, resize_factor=3):
    '''
    Utility function that draw locs as circles on the given image im.
    '''
    im = cv2.resize(im, (0,0), fx=resize_factor, fy=resize_factor)
    if len(im.shape) < 3:
        im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    for loc in locs:
        cv2.circle(im, (loc[0]*resize_factor, loc[1]*resize_factor), 2, (0,0,255), -1)
    cv2.imshow('Interest Points', im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()
    return im

def main():
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    '''for submission'''
    # displayPyramid(im_pyr)

    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    '''for submission'''
    # displayPyramid(DoG_pyr)

    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)

    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    # Draw the result keypoints as circles on the image
    '''for submission'''
    # drawn = drawPoints(im, locsDoG, resize_factor=3)

if __name__ == '__main__':
    main()
