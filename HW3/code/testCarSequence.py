import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

from LucasKanade import LucasKanadeTracker

# '''[p1:p3, p0:p2]'''
INPUT_NPY = "../data/carseq.npy"
INIT_RECT = [[59], [116], [145], [151]]
IMG_SAVE_NAME = "carseqrects"
RESULTS_SAVE_PATH = "../results/carseqrects.npy"

'''
Given a video and the tracking results, visualize the results using animation
'''
def visualizeTracker(frames, rects_list, img_save_name, save_frame=[0,99,199,299,399], colors=['y', 'g']):
    # Creating figure and ax
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #do nothing
    def init():
        pass

    '''
    function for updating one frame
    '''
    def animate(i):
        ax.clear()
        It = frames[:,:,i]
        # Convert the grayscale to RGB for display
        show = np.stack([It] * 3, axis=2)
        show = (show*255).astype(np.uint8)
        ax.imshow(show)

        rects_to_draw = []
        for j, rects in enumerate(rects_list):
            c = colors[j]
            x1, y1, x2, y2 = rects[i]
            # Create a Rectangle patch
            r = patches.Rectangle((x1,y1),(x2-x1),(y2-y1),linewidth=1,edgecolor=c,facecolor='none')

            rects_to_draw.append(r)

        # Add the patches to the Axes
        for r in rects_to_draw[::-1]:
            ax.add_patch(r)

        ax.set_xticks([])
        ax.set_yticks([])

        if i in save_frame:
            print("Saving ../results/%s_%d.png" % (img_save_name, i))
            fig.savefig("../results/%s_%d.png" % (img_save_name, i), bbox_inches='tight')

        # Close the window after animation
        if i == frames.shape[2]-1:
            plt.close(fig)

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=30, frames=frames.shape[2], repeat=False)
    plt.show()

'''
Wrapper function to run a given tracker and return the result rectangles
'''
def runTracker(frames, init_rect, tracker):
    results = []
    results.append(init_rect)

    for i in range(frames.shape[2] - 1):
        It = frames[:,:,i]
        It1 = frames[:,:,i+1]

        tracker.update(It, It1)
        rect = tracker.get_rect()
        results.append(rect)

    results = np.stack(results, axis=0).squeeze()
    return results

def main():
    frames = np.load(INPUT_NPY)
    frame0 = frames[:,:,0]
    rect0 = np.array(INIT_RECT, dtype=float)

    tracker = LucasKanadeTracker(rect0)

    print("Running LucasKanadeTracker", end="...")
    results = runTracker(frames, rect0, tracker)
    print("Done")

    print("Saving result of LucasKanadeTracker to:", RESULTS_SAVE_PATH)
    np.save(RESULTS_SAVE_PATH, results)

    print("Visualizing results of LucasKanadeTracker...")
    visualizeTracker(frames, [results], IMG_SAVE_NAME)
    print("Visualization Done")

if __name__ == '__main__':
    main()
