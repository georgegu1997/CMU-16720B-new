import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

dim = 32
# do PCA
##########################
##### your code here #####
U, S, Vh = np.linalg.svd(train_x.T)
##########################

# rebuild a low-rank version
lrank = None
##########################
##### your code here #####
X = U[:, :dim] # principle components
P = X.T # projection matrix
C = np.diag(S[:dim]).dot(Vh[:dim]) # low-rank representation
lrank = C.T
print("projection matrix:", P.shape)
print("low rank representation:", lrank.shape)
##########################

# rebuild it
recon = None
##########################
##### your code here #####
recon_train_x = X.dot(C)
recon = recon_train_x.T
print("recon_train:", recon.shape)
##########################

# build valid dataset
recon_valid = None
##########################
##### your code here #####
lrank_valid_x = P.dot(valid_x.T)
recon_valid_x = X.dot(lrank_valid_x)
recon_valid = recon_valid_x.T
print("recon_valid:", recon_valid.shape)
##########################

# visualize the comparison and compute PSNR
##########################
##### your code here #####
np.random.seed(2020)
selected_classes = np.random.choice(np.arange(36), 5)
fig, axes = plt.subplots(nrows=5, ncols=4)
for i, c in enumerate(selected_classes):
    # print((valid_y[:, c] == 1).sum())
    x_c = valid_x[valid_y[:, c] == 1]
    selected_x = x_c[np.random.choice(np.arange(x_c.shape[0]), 2)]
    recon_x = X.dot(P.dot(selected_x.T)).T
    axes[i, 0].imshow(selected_x[0].reshape((32,32)).T, cmap="gray")
    axes[i, 1].imshow(recon_x[0].reshape((32,32)).T, cmap="gray")
    axes[i, 2].imshow(selected_x[1].reshape((32,32)).T, cmap="gray")
    axes[i, 3].imshow(recon_x[1].reshape((32,32)).T, cmap="gray")

axes[0, 0].set_title("input")
axes[0, 1].set_title("recon")
axes[0, 2].set_title("input")
axes[0, 3].set_title("recon")
for ax in axes.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
##########################

# evaluate PSNR
##########################
##### your code here #####
total_psnr = 0
for i in range(valid_x.shape[0]):
    im = valid_x[i].reshape((32, 32))
    recon_im = recon_valid[i].reshape((32, 32))
    total_psnr += psnr(im, recon_im)
avg_psnr = total_psnr / valid_x.shape[0]
print("Averge PSNR:", avg_psnr)
##########################
