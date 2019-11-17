import numpy as np
import scipy.io
from nn import *
from collections import Counter

np.random.seed(2020)

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
class AutoEncoder():
    def __init__(self, params={}, input_size=1024, hidden_size=32, output_size=1024):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.params = params
        if len(params) > 0:
            # already initialized, then keep the params
            pass
        else:
            initialize_weights(input_size, hidden_size, self.params, "input")
            initialize_weights(hidden_size, hidden_size, self.params, "hidden1")
            initialize_weights(hidden_size, hidden_size, self.params, "hidden2")
            initialize_weights(hidden_size, output_size, self.params, "output")

    def forward(self, x):
        h1 = forward(x, self.params, "input", relu)
        h2 = forward(h1, self.params, "hidden1", relu)
        h3 = forward(h2, self.params, "hidden2", relu)
        out = forward(h3, self.params, "output", sigmoid)
        return out

    def backward(self, dout):
        dh3 = backwards(dout, self.params, "output", sigmoid_deriv)
        dh2 = backwards(dh3, self.params, "hidden2", relu_deriv)
        dh1 = backwards(dh2, self.params, "hidden1", relu_deriv)
        dx = backwards(dh1, self.params, "input", relu_deriv)
        return dx

def totalSquaredError(y, out):
    diff = (out-y)
    diffsq = diff**2
    loss = diffsq.sum()
    cache = (out, y, diff, diffsq)
    return loss, cache

def totalSquaredErrorBackward(loss, cache):
    (out, y, diff, diffsq) = cache
    dout = 2*(out-y)
    return dout

def sgdOptimize(params, learning_rate):
    for k,v in sorted(list(params.items())):
        if 'grad' in k:
            name = k.split('_')[1]
            # print(np.linalg.norm(learning_rate * v))
            params[name] -= learning_rate * v
    return params

def momentumOptimize(params, learning_rate):
    for k,v in sorted(list(params.items())):
        if 'grad' in k:
            name = k.split('_')[1]
            params["m_"+name] = 0.9*params["m_"+name] - learning_rate*v
            params[name] += params["m_"+name]
    return params

input_size = 1024
output_size = 1024
ae = AutoEncoder(params, input_size=1024, hidden_size=32, output_size=1024)
##########################

# should look like your previous training loops
epoch_list = []
train_loss_list = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        # forward
        out = ae.forward(xb)
        loss, error_cache = totalSquaredError(xb, out)
        total_loss += loss

        # backward
        dout = totalSquaredErrorBackward(loss, error_cache)
        dx = ae.backward(dout)

        # apply gradient
        ae.params = momentumOptimize(ae.params, learning_rate)
        ##########################

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
        epoch_list.append(itr+1)
        train_loss_list.append(total_loss)
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

params = ae.params

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results

# # Plot the training losses
# plt.plot(epoch_list, train_loss_list)
# plt.title("Training loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

##########################
##### your code here #####
np.random.seed(2020)
selected_classes = np.random.choice(np.arange(36), 5)
fig, axes = plt.subplots(nrows=5, ncols=4)
for i, c in enumerate(selected_classes):
    # print((valid_y[:, c] == 1).sum())
    x_c = valid_x[valid_y[:, c] == 1]
    selected_x = x_c[np.random.choice(np.arange(x_c.shape[0]), 2)]
    reconstruct = ae.forward(selected_x)
    axes[i, 0].imshow(selected_x[0].reshape((32,32)).T, cmap="gray")
    axes[i, 1].imshow(reconstruct[0].reshape((32,32)).T, cmap="gray")
    axes[i, 2].imshow(selected_x[1].reshape((32,32)).T, cmap="gray")
    axes[i, 3].imshow(reconstruct[1].reshape((32,32)).T, cmap="gray")

axes[0, 0].set_title("input")
axes[0, 1].set_title("output")
axes[0, 2].set_title("input")
axes[0, 3].set_title("output")
for ax in axes.reshape(-1):
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
##########################

# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
total_psnr = 0
for x in valid_x:
    recon = ae.forward(x)
    im = x.reshape((32, 32))
    recon_im = recon.reshape((32, 32))
    total_psnr += psnr(im, recon_im)
avg_psnr = total_psnr / valid_x.shape[0]
print("Averge PSNR:", avg_psnr)
##########################
