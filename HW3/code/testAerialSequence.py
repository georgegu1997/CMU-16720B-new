import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanadeAffine import LucasKanadeAffine
from SubtractDominantMotion import SubtractDominantMotion
from scipy.ndimage.morphology import binary_dilation
# import cv2

INPUT_NPY = "../data/aerialseq.npy"
IMG_SAVE_NAME = "aerialseq"
SAVE_FRAME = [29,59,89,119]

'''
image1 is the background and image2 is the foreground
'''
def alphaBlend(image1, image2, alpha):
    im1 = image1.astype(float)
    im2 = image2.astype(float)
    mask = im2.sum(axis=2) > 0
    im = im1.copy()
    im[mask] = im1[mask] * (1-alpha) + im2[mask] * alpha
    return im.clip(0,255).round().astype(np.uint8)

def visualizeMotion(frames, masks, img_save_name, save_frame=[29,59,89,119]):
    # Creating figure and ax
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #do nothing
    def init():
        pass

    # function for updating one frame
    def animate(i):
        ax.clear()
        It = frames[:,:,i]
        # Convert the grayscale to RGB for display
        show = np.stack([It] * 3, axis=2)
        show = (show*255).astype(np.uint8)

        dilation_structure = np.ones((3, 3))
        # No motion information for the first frame
        if i > 0:
            mask = masks[:,:,i].copy()
            mask = binary_dilation(mask, dilation_structure)

            red_mask = np.zeros((*mask.shape,3))
            red_mask[:,:,2] = (mask*255).astype(np.uint8)
            show = alphaBlend(show, red_mask, 0.5)

        ax.imshow(show)

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
    return

def main():
    frames = np.load(INPUT_NPY)
    print(frames.shape)

    # Get the mask for all the frame
    masks = [np.zeros(frames.shape[:2])]
    for i in range(frames.shape[2] - 1):
    # for i in range(20):
        It = frames[:,:,i]
        It1 = frames[:,:,i+1]

        M = LucasKanadeAffine(It, It1)

        mask = SubtractDominantMotion(It, It1)
        masks.append(mask)

    masks = np.stack(masks, axis=-1)
    print(masks.shape)

    masks_test = masks.copy()

    masks_save = masks[:,:,[29,59,89,119]]
    print("Saving masks_save:", masks_save.shape)
    np.save("../results/aerialseqmasks.npy", masks_save)

    visualizeMotion(frames, masks, img_save_name=IMG_SAVE_NAME, save_frame=SAVE_FRAME)


if __name__ == "__main__":
    main()
