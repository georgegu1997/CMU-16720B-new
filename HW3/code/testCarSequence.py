import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

from LucasKanade import LucasKanadeTracker
import cv2

# '''[p1:p3, p0:p2]'''
INPUT_NPY = "../data/carseq.npy"
INIT_RECT = [[59], [116], [145], [151]]
IMG_SAVE_NAME = "carseqrects"
RESULTS_SAVE_PATH = "../results/carseqrects.npy"

def runTracker(frames, init_rect, trackers, img_save_name, results_save_path=None, to_show=False, save_frame=[0, 99, 199, 299, 399]):
    COLORS = [[0,255,255], [0,255,0]]
    results = []

    for i in range(frames.shape[2] - 1):
        It = frames[:,:,i]
        It1 = frames[:,:,i+1]

        # Convert gray image to RGB one
        if to_show or i in save_frame:
            show = cv2.cvtColor((It1*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)

        for j, tracker in enumerate(trackers):
            tracker.update(It, It1)
            rect = tracker.get_rect()

            # Only save the results from the last tracker
            if j == len(trackers) - 1 :
                results.append(rect)

            # Draw the rectangle for each tracker
            if to_show or i in save_frame:
                cv2.rectangle(show, (rect[0], rect[1]), (rect[2], rect[3]), COLORS[j], 2)

        if to_show:
            cv2.imshow("", show)
            cv2.waitKey(0) # press any key to exit
        if i in save_frame:
            print("Saving ../results/%s_%d.png" % (img_save_name, i))
            cv2.imwrite("../results/%s_%d.png" % (img_save_name, i), show)

    if to_show:
        cv2.destroyAllWindows()

    results = np.array(results)
    if not results_save_path is None:
        print("Saveing", results_save_path)
        np.save(results_save_path, results)

    return results

def main():
    frames = np.load(INPUT_NPY)
    frame0 = frames[:,:,0]
    rect0 = np.array(INIT_RECT, dtype=float)

    trackers = []
    trackers.append(LucasKanadeTracker(rect0))
    results = runTracker(frames, INIT_RECT, trackers, IMG_SAVE_NAME, RESULTS_SAVE_PATH, to_show=True)

if __name__ == '__main__':
    main()
