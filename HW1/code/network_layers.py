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
	output_dim = weight.shape[0]
	# transpose to (output_dim, kernel_size, kernel_size, input_dim)
	weight = weight.transpose((0, 2, 3, 1))

	feat = []
	print("x.shape:", x.shape)
	for o in range(output_dim):
		feat.append(scipy.ndimage.convolve(x, weight[o], mode='constant'))
		print("feat.shape:", feat.shape)

	feat = np.concatenate(feat, axis = 0)

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
	
	pass

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
