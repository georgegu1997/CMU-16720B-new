import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanadeTrackerWithTemplateCorrection, LucasKanadeTracker
from testCarSequence import runTracker, visualizeTracker

INPUT_NPY = "../data/carseq.npy"
INIT_RECT = [[59], [116], [145], [151]]
IMG_SAVE_NAME = "carseqrects-wcrt"
RESULTS_SAVE_PATH = "../results/carseqrects-wcrt.npy"

def main():
    frames = np.load(INPUT_NPY)
    frame0 = frames[:,:,0]
    rect0 = np.array(INIT_RECT, dtype=float)

    LK_tracker = LucasKanadeTracker(rect0)
    LKTR_tracker = LucasKanadeTrackerWithTemplateCorrection(rect0, frame0)

    print("Running LucasKanadeTracker", end="...")
    LK_results = runTracker(frames, rect0, LK_tracker)
    print("Done")

    print("Running LucasKanadeTrackerWithTemplateCorrection", end="...")
    LKTR_results = runTracker(frames, rect0, LKTR_tracker)
    print("Done")

    print("Saving result of LucasKanadeTrackerWithTemplateCorrection to:", RESULTS_SAVE_PATH)
    print(LKTR_results.shape)
    np.save(RESULTS_SAVE_PATH, LKTR_results)

    print("Visualizing results of LucasKanadeTracker...")
    visualizeTracker(frames, [LKTR_results, LK_results], IMG_SAVE_NAME)
    print("Visualization Done")

if __name__ == '__main__':
    main()
