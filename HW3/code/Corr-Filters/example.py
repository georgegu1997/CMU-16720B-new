import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.patches as patches

'''For matplotlib version over 0.99, please use/uncomment the following line'''
from mpl_toolkits.mplot3d import axes3d, Axes3D

from scipy.ndimage import convolve, correlate

if __name__ == "__main__":
    img = np.load('lena.npy')

    # template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
    pts = np.array([[248, 292, 248, 292],
                    [252, 252, 280, 280]])

    # size of the template (h, w)
    dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                      pts[0, 1] - pts[0, 0] + 1])

    # set template corners
    tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                          [0, 0, dsize[0] - 1, dsize[0] - 1]])


    # apply warp p to template region of img
    def imwarp(p):
        global img, dsize
        return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]


    # get positive example
    gnd_p = np.array([252, 248])  # ground truth warp
    x = imwarp(gnd_p)  # the template

    # stet up figure
    fig, axarr = plt.subplots(1, 3)
    axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
    patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0],
                              linewidth=1, edgecolor='r', facecolor='none')
    axarr[0].add_patch(patch)
    axarr[0].set_title('Image')

    cropax = axarr[1].imshow(x, cmap=plt.get_cmap('gray'))
    axarr[1].set_title('Cropped Image')

    dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
    dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
    [dpx, dpy] = np.meshgrid(dx, dy)
    dpx = dpx.reshape(-1, 1)
    dpy = dpy.reshape(-1, 1)
    dp = np.hstack((dpx, dpy))
    N = dpx.size

    all_patches = np.ones((N*dsize[0], dsize[1]))
    all_patchax = axarr[2].imshow(all_patches, cmap=plt.get_cmap('gray'),
                                  aspect='auto', norm=colors.NoNorm())
    axarr[2].set_title('Concatenation of Sub-Images (X)')

    X = np.zeros((N, N))
    Y = np.zeros((N, 1))

    sigma = 5


    def init():
        return [cropax, patch, all_patchax]


    def animate(i):
        global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

        if i < N:  # If the animation is still running
            xn = imwarp(dp[i, :] + gnd_p)
            X[:, i] = xn.reshape(-1)
            Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
            all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
            cropax.set_data(xn)
            all_patchax.set_data(all_patches.copy())
            all_patchax.autoscale()
            patch.set_xy(dp[i, :] + gnd_p)
            return [cropax, patch, all_patchax]
        else:  # Stuff to do after the animation ends
            fig3d = plt.figure()

            '''Use the following line for matplotlib version over 1.00'''
            ax3d = Axes3D(fig3d)
            '''Use the following line for matplotlib version under 0.99'''
            # ax3d = fig3d.add_subplot(111, projection='3d')

            ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize),
                              Y.reshape(dsize), cmap=plt.get_cmap('coolwarm'))

            # Place your solution code for question 4.3 here
            plt.show()

            # Solve for the filter g
            S = X.dot(X.T)
            XY = X.dot(Y)

            plt.figure()
            # The template
            plt.imshow(x, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_gt.png")

            # lambda = 0
            g = np.linalg.inv(S).dot(XY)
            g = g.reshape(dsize)
            plt.imshow(g, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_lambda_0.png")

            # correlate with the image
            corr = correlate(img, g)
            plt.imshow(corr, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_lambda_0_corr.png")

            # convolve with the image
            conv = convolve(img, g)
            plt.imshow(conv, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_lambda_0_conv.png")

            # lambda = 1
            g = np.linalg.inv(S+1*np.eye(S.shape[0])).dot(XY)
            g = g.reshape(dsize)
            plt.imshow(g, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_lambda_1.png")

            # correlate with the image
            corr = correlate(img, g)
            plt.imshow(corr, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_lambda_1_corr.png")
            # plt.show()

            # convolve with the image
            conv = convolve(img, g)
            plt.imshow(conv, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_lambda_1_conv.png")

            # convolve the image with the flipped filter
            conv_flip = convolve(img, np.flip(g, (0,1)))
            plt.imshow(conv_flip, cmap=plt.get_cmap('gray'))
            plt.savefig("../../results/q4_3_lambda_1_conv_flip.png")

            return []


    # Start the animation
    ani = animation.FuncAnimation(fig, animate, frames=N+1,
                                  init_func=init, blit=True,
                                  repeat=False, interval=10)
    plt.show()
