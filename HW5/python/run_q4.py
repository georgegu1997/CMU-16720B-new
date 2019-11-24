import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

GROUND_TRUTH = [
    [
    "TODOLIST",
    "1MAKEATODOLIST",
    "2CHECKOFFTHEFIRST",
    "THINGONTODOLIST",
    "3REALIZEYOUHAVEALREADY",
    "COMPLETED2THINGS",
    "4REWARDYOURSELFWITH",
    "ANAP",
    ],
    [
    "ABCDEFG",
    "HIJKLMN",
    "OPQRSTU",
    "VWXYZ",
    "1234567890",
    ],
    [
    "HAIKUSAREEASY",
    "BUTSOMETIMESTHRYDONTMAKESENSE",
    "REFRIGERATOR",
    ],
    [
    "DEEPLEARNING",
    "DEEPERLEARNING",
    "DEEPESTLEARNING",
    ]
]

'''
Hierarchical clustering method on y coordinates of the bboxes to find rows in detection results
    Specifically, Single Linkage clustering
'''
def cluster(bboxes):
    X = bboxes[:, 0]
    N = bboxes.shape[0]
    dists = np.absolute(X.reshape((-1, 1)) - X.reshape((1, -1)))
    # Only get the upper diagonal part
    dists[np.tril(np.ones_like(dists)).astype(np.bool)] = 1e10

    # Combine the distances with their corresponding positions
    xv, yv = np.meshgrid(np.arange(N), np.arange(N))
    dists = np.stack([dists, xv, yv], axis=2)
    dists = dists.reshape((-1, 3))

    C = np.arange(N) # label of the cluster numbers
    # Change all labels of the same cluster with idx2 to the label of idx1
    def mergeClass(C, idx1, idx2):
        label1, label2 = C[idx1], C[idx2]
        C[C == label2] = label1
        return C

    # Sort the list according to distances
    sort_dists = dists[dists[:, 0].argsort()]

    # If the distances is smaller than this, merge them anyway
    safe_dist = 15
    # If the
    # Keep track of the last distance of merge, if the next merge distance is to large
    # than merge_rate times of this, stop and return
    last_dist = 0
    merge_rate = 1.2
    for d in sort_dists:
        if d[0] > 5e9:
            break
        if d[0] <= safe_dist:
            C = mergeClass(C, d[1], d[2])
        elif d[0] <= last_dist * merge_rate:
            C = mergeClass(C, d[1], d[2])
        else:
            break
        last_dist = d[0]

    # Rename the classes
    for i, v in enumerate(np.unique(C)):
        C[C==v] = i
    return C

def evaluateOCRResults(text_by_line, gt):
    correct_num, total_num = 0.0, 0.0
    for line, line_gt in zip(text_by_line, gt):
        print(line)
        # print(line_gt)
        for l, lgt in zip(line, line_gt):
            if l == lgt:
                correct_num +=1
            total_num += 1
    print("Accuracy: %.3f" % (correct_num / total_num))

def main():
    for img_i, img in enumerate(os.listdir('../images')):
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
        bboxes, bw = findLetters(im1)

        plt.imshow(bw, cmap="gray")
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
        plt.show()
        # find the rows using..RANSAC, counting, clustering, etc.
        ##########################
        ##### your code here #####
        # # Sort the bounding box by their x coordinate
        # bboxes = bboxes[bboxes[:, 1].argsort()]

        # Cluster
        classes = cluster(bboxes)
        # Group the bboxes by class
        line_labels = np.unique(classes)
        line_idx = []
        for label in line_labels:
            this_line_idx = np.where(classes == label)[0]
            # Sort characters by x coordinates
            this_line_idx = this_line_idx[bboxes[this_line_idx, 1].argsort()]
            line_idx.append(this_line_idx)
        # Sort lines by the first y index
        first_ys = np.array([bboxes[line[0], 0] for line in line_idx])
        sorted_line_idx = []
        for i in first_ys.argsort():
            sorted_line_idx.append(line_idx[i])
        line_idx = sorted_line_idx
        ##########################

        # crop the bounding boxes
        # note.. before you flatten, transpose the image (that's how the dataset is!)
        # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
        ##########################
        ##### your code here #####
        X = []
        pad_width = 5
        for box in bboxes:
            y1, x1, y2, x2 = box
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            l = max(y2-y1, x2-x1)
            nx1, nx2 = int(cx-l/2), int(cx+l/2)
            ny1, ny2 = int(cy-l/2), int(cy+l/2)
            crop = bw[ny1:ny2, nx1:nx2].copy()
            crop = skimage.transform.resize(crop.astype(float), (32-pad_width*2, 32-pad_width*2))
            crop = 1 - (crop < 0.9)
            crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=1)
            # plt.imshow(crop, cmap='gray')
            # plt.show()
            X.append(crop.T.reshape(-1))
        X = np.array(X)
        ##########################

        # load the weights
        # run the crops through your neural network and print them out
        import pickle
        import string
        letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
        params = pickle.load(open('q3_weights.pickle','rb'))
        ##########################
        ##### your code here #####
        # Forward
        h1 = forward(X, params, 'layer1') # First layer
        probs = forward(h1, params, 'output', softmax) # Second layer
        pred_label = probs.argmax(axis=1)

        chars = string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)])
        text_by_line = []
        for r in line_idx:
            line = ""
            for idx in r:
                line += chars[pred_label[int(idx)]]
            text_by_line.append(line)

        print()
        print("Image:", img)
        evaluateOCRResults(text_by_line, GROUND_TRUTH[img_i])
        ##########################

if __name__ == '__main__':
    main()
