import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
        patch_width - the width of the image patch (usually 9)
        nbits       - the number of tests n in the BRIEF descriptor

    OUTPUTS
        compareX and compareY - LINEAR indices into the patch_width x patch_width image
                                patch and are each (nbits,) vectors.
    '''

    #############################
    # TO DO ...
    # Generate testpattern here
    # Implement the second method from the paper: the isotropic Guassian distribution
    sigma = patch_width / 5.0
    mu = (patch_width-1)/2
    compareX = np.random.normal(mu, sigma, (nbits, 2)).round().clip(0, patch_width-1)
    compareY = np.random.normal(mu, sigma, (nbits, 2)).round().clip(0, patch_width-1)
    # Convert them to linear coordinates
    compareX = (compareX[:,0] * patch_width + compareX[:,1]).astype(int)
    compareY = (compareY[:,0] * patch_width + compareY[:,1]).astype(int)
    return  compareX, compareY


# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])


def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute brief feature
    INPUT
        locsDoG - locsDoG are the keypoint locations returned by the DoG
                detector.
        levels  - Gaussian scale levels that were given in Section1.
        compareX and compareY - linear indices into the
                                (patch_width x patch_width) image patch and are
                                each (nbits,) vectors.


    OUTPUT
        locs - an m x 3 vector, where the first two columns are the image
                coordinates of keypoints and the third column is the pyramid
                level of the keypoints.
        desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
                of valid descriptors in the image and will vary.
    '''

    ##############################
    # TO DO ...
    # compute locs, desc here
    patch_width = 9

    # Smooth the image
    im = cv2.GaussianBlur(im, (0,0), np.sqrt(2))

    # Exclude the locations where we cannot get full patches
    h, w = im.shape
    locs = locsDoG.copy()
    # x coordinate corresponds to columns and y coordinate corresponds to rows
    id_x = np.logical_and(locs[:, 0] > (patch_width-1)/2, locs[:, 0] < w - (patch_width-1)/2)
    id_y = np.logical_and(locs[:, 1] > (patch_width-1)/2, locs[:, 1] < h - (patch_width-1)/2)
    locs = locs[np.logical_and(id_x, id_y)]

    # Compute brief for each patch
    desc = []
    for loc in locs:
        x, y, _ = loc
        # x coordinate corresponds to columns and y coordinate corresponds to rows
        p = im[int(y-(patch_width-1)/2):int(y+(patch_width+1)/2), \
                int(x-(patch_width-1)/2):int(x+(patch_width+1)/2)]
        p = p.reshape(-1)
        bits = p[compareX] < p[compareY]
        desc.append(bits)
    desc = np.array(desc)

    return locs, desc


def briefLite(im):
    '''
    INPUTS
        im - gray image with values between 0 and 1

    OUTPUTS
        locs - an m x 3 vector, where the first two columns are the image coordinates
            of keypoints and the third column is the pyramid level of the keypoints
        desc - an m x n bits matrix of stacked BRIEF descriptors.
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''

    ###################
    # TO DO ...

    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255

    locsDoG, gaussian_pyramid = DoGdetector(im)
    locs,desc = computeBrief(im, gaussian_pyramid, locsDoG, np.sqrt(2), [-1,0,1,2,3,4], compareX, compareY)
    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''

    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r', linewidth=0.3)
        plt.plot(x,y,'g.', ms=2)
    # plt.show()

def matchImages(file1, file2, save_file = None):
    im1 = cv2.imread(file1)
    im2 = cv2.imread(file2)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    if not save_file is None:
        plotMatches(im1,im2,matches,locs1,locs2)
        plt.savefig(save_file, dpi=300)
    else:
        plotMatches(im1,im2,matches,locs1,locs2)

def main():
    # test makeTestPattern
    compareX, compareY = makeTestPattern()

    # test briefLite
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs, desc = briefLite(im)
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)

    matchImages('../data/model_chickenbroth.jpg', '../data/chickenbroth_01.jpg', "../results/chickenbroth_01_match.jpg")
    matchImages('../data/incline_L.png', '../data/incline_R.png', "../results/incline_match.jpg")
    matchImages('../data/pf_scan_scaled.jpg', '../data/pf_stand.jpg', "../results/pf_stand_match.jpg")
    matchImages('../data/pf_scan_scaled.jpg', '../data/pf_floor.jpg', "../results/pf_floor_match.jpg")
    matchImages('../data/pf_scan_scaled.jpg', '../data/pf_floor_rot.jpg', "../results/pf_floor_rot_match.jpg")
    matchImages('../data/pf_scan_scaled.jpg', '../data/pf_desk.jpg', "../results/pf_desk_match.jpg")
    matchImages('../data/pf_scan_scaled.jpg', '../data/pf_pile.jpg', "../results/pf_pile_match.jpg")


if __name__ == '__main__':
    main()
