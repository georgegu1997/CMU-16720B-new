import numpy as np
import scipy.ndimage
import os

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: list of shape (L, 3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''

    pass


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''

    H, W, input_dim = x.shape
    output_dim = weight.shape[0]
    # transpose to (output_dim, kernel_size, kernel_size, input_dim)
    weight = weight.transpose((0, 2, 3, 1))

    feat = []
    print("x.shape:", x.shape)
    for o in range(output_dim):
        f = []
        for i in range(input_dim):
            f.append( scipy.ndimage.correlate(x[:,:,i], weight[o,:,:,i], mode='constant') )
        f = np.stack(f, axis=2)
        f = f.sum(axis = 2)
        feat.append(f)

    feat = np.stack(feat, axis = 2)
    feat += bias.reshape((1,1,output_dim))

    print("feat shape after concatenate;", feat.shape)

    return feat


def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''
    y = np.maximum(x, 0)

    return y

# References:
# https://ipython-books.github.io/46-using-stride-tricks-with-numpy/
# https://ipython-books.github.io/47-implementing-an-efficient-rolling-average-algorithm-with-stride-tricks/
# The max pooling used by vgg16 is MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''
    H, W, input_dim = x.shape
    H_out = H//size
    W_out = W//size

    # Sliding window using stride tricks
    xy_strides = np.array(x.strides)[:2] * size
    new_strides = tuple(xy_strides) + x.strides # Tuple addition = concatenation
    new_shape = (H_out, W_out, size, size, input_dim)
    rolling_x = np.lib.stride_tricks.as_strided(x, shape=new_shape, strides=new_strides)

    y = np.amax(rolling_x, axis=(2, 3))

    return y

def linear(x,W,b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    y = W.dot(x) + b
    return y
