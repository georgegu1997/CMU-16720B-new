import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    # Denoising
    image = skimage.restoration.denoise_wavelet(image, multichannel=True, convert2ycbcr=True)

    # Thresholding
    image = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(image)
    image = image < thresh

    # Morphology
    image = skimage.morphology.binary_closing(image, np.ones((7,8)))
    bw = (1-image.copy())

    # Label by connectivity
    image, n_label = skimage.measure.label(image, connectivity=2, return_num=True)

    # Get the region for each character
    bboxes = []
    for i in range(n_label+1):
        # idx = np.array((image == i).nonzero())
        idx = np.array(np.where(image == i))
        idx_max = idx.max(axis=1)
        idx_min = idx.min(axis=1)
        y1, x1, y2, x2 = idx_min[0], idx_min[1], idx_max[0], idx_max[1]
        # Discard the detection with small y span or small size
        if (y2 - y1) < 25 or (y2 - y1) + (x2 - x1) < 20:
            continue
        bbox = [y1, x1, y2, x2]
        bboxes.append(bbox)

    # import matplotlib.pyplot as plt
    # plt.subplot(111)
    # plt.imshow(image.astype(float))
    # plt.show()

    ##########################

    return bboxes, bw
