import numpy as np
import cv2
import os
from BRIEF import briefLite, briefMatch, plotMatches

import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def main():
    im = cv2.imread('../data/model_chickenbroth.jpg')
    # Enlarge the image for safe rotation
    diagonal = int((im.shape[0]**2 + im.shape[1]**2) ** (1/2))
    im1 = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
    im1[diagonal//2-im.shape[0]//2:diagonal//2-im.shape[0]//2+im.shape[0], \
        diagonal//2-im.shape[1]//2:diagonal//2-im.shape[1]//2+im.shape[1], :] = im
    image_center = tuple(np.array(im1.shape[1::-1]) / 2)

    results = []
    for i in range(36):
        # Rotate the image
        angle = 10*i
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        im2 = cv2.warpAffine(im1, rot_mat, im1.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Do the BRIEF matching
        locs1, desc1 = briefLite(im1)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1, desc2)
        n_matches = matches.shape[0]

        results.append([angle, n_matches])

    results = np.array(results)
    ax = plt.subplot()
    ax.bar(results[:,0] - 2, results[:,1], width=2, color='b', align='center', label="matches")
    ax.set_xlabel("Rotation (degree)")
    ax.set_ylabel("Number of matches")
    '''for submission'''
    # plt.show()

if __name__ == "__main__":
    main()
