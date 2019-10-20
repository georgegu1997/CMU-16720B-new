import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanadeAffine import LucasKanadeAffine
from SubtractDominantMotion import SubtractDominantMotion
from scipy.ndimage.morphology import binary_dilation
import cv2

INPUT_NPY = "../data/aerialseq.npy"
IMG_SAVE_NAME = "aerialseq"
SAVE_FRAME = [29,59,89,119]

def alphaBlend(image1, image2, alpha):
    im1 = image1.astype(float)
    im2 = image2.astype(float)
    mask = im2.sum(axis=2) > 0
    im = im1.copy()
    im[mask] = im1[mask] * alpha + im2[mask] * (1-alpha)
    return im.clip(0,255).round().astype(np.uint8)

def main():
    frames = np.load(INPUT_NPY)

    # Get the mask for all the frame
    masks = []
    for i in range(frames.shape[2] - 1):
        It = frames[:,:,i]
        It1 = frames[:,:,i+1]

        M = LucasKanadeAffine(It, It1)

        mask = SubtractDominantMotion(It, It1)
        masks.append(mask)

        

        # dilation_structure = np.ones((5,5))
        # mask = binary_dilation(mask, dilation_structure)

        # red_mask = np.zeros((*mask.shape,3))
        # red_mask[:,:,2] = (mask*255).astype(np.uint8)

        # show = cv2.cvtColor((It1*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # show = alphaBlend(show, red_mask, 0.6)

        # if i in SAVE_FRAME:
        #     print("Saving ../results/%s_%d.png" % (IMG_SAVE_NAME, i))
        #     cv2.imwrite("../results/%s_%d.png" % (IMG_SAVE_NAME, i), show)

    #     cv2.imshow("", show)
    #     cv2.waitKey(0) # press any key to exit
    #
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
