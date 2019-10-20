import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanadeTracker
from LucasKanadeBasis import LucasKanadeTrackerBasis
from testCarSequence import runTracker, visualizeTracker

INPUT_NPY = "../data/sylvseq.npy"
INIT_RECT = [[101], [61], [155], [107]]
IMG_SAVE_NAME = "sylvseqrects"
RESULTS_SAVE_PATH = "../results/sylvseqrects.npy"
INPUT_BASES = "../data/sylvbases.npy"

def main():
    frames = np.load(INPUT_NPY)
    bases = np.load(INPUT_BASES)
    frame0 = frames[:,:,0]
    rect0 = np.array(INIT_RECT, dtype=float)

    LK_tracker = LucasKanadeTracker(rect0)
    LKB_tracker = LucasKanadeTrackerBasis(rect0, bases=bases)

    print("Running LucasKanadeTracker", end="...")
    LK_results = runTracker(frames, rect0, LK_tracker)
    print("Done")

    print("Running LucasKanadeTrackerBasis", end="...")
    LKB_results = runTracker(frames, rect0, LKB_tracker)
    print("Done")

    print("Saving result of LucasKanadeTrackerBasis to:", RESULTS_SAVE_PATH)
    print(LKB_results.shape)
    np.save(RESULTS_SAVE_PATH, LKB_results)

    print("Visualizing results of LucasKanadeTracker...")
    visualizeTracker(frames, [LKB_results, LK_results], IMG_SAVE_NAME, save_frame=[0,199,299,349,399])
    print("Visualization Done")

if __name__ == "__main__":
    main()
