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
class PCA():
    def __init__(self):
        pass

    '''A is of shape (num datapoints, num features)'''
    def train(self, A, dim):
        self.dim = dim
        A = A.T
        self.mean = A.mean(axis=1, keepdims=True)
        A -= self.mean
        U, S, Vh = np.linalg.svd(A)
        self.X = U[:, :dim] # principle components
        self.P = self.X.T # projection matrix

    def project(self, A):
        A = A.T
        lrank = self.P.dot(A - self.mean)
        return lrank.T

    '''A is of shape (num features, num datapoints)'''
    def projectAndRecon(self, A):
        A = A.T
        lrank = self.P.dot(A - self.mean)
        recon = self.X.dot(lrank)
        recon += self.mean
        return recon.T

pca = PCA()
pca.train(train_x, dim=dim)
##########################

# rebuild a low-rank version
lrank = None
##########################
##### your code here #####
print("projection matrix:", pca.P.shape)
lrank = pca.project(train_x)
print("low rank representation:", lrank.shape)
##########################

# rebuild it
recon = None
##########################
##### your code here #####
recon = pca.projectAndRecon(train_x)
print("recon_train:", recon.shape)
##########################

# build valid dataset
recon_valid = None
##########################
##### your code here #####
recon_valid = pca.projectAndRecon(valid_x)
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
    recon_x = pca.projectAndRecon(selected_x)
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
