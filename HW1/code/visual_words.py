import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random

FILTER_SCALES = [1, 2, 4, 8, 8*np.sqrt(2)]

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # ----- TODO -----
    # Convert the image to float type between 0 and 1
    print("image.dtype:", image.dtype)
    if image.dtype == np.uint8 or image.max() > 10:
        image = image.astype(float) / 255.0
    # Check and convert the dimensions
    print("input image shape:", image.shape)
    if image.ndim = 2:
        image = np.stack([image]*3, axis=2)



    return filter_responses

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----

    pass


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''


    i, alpha, image_path = args
    # ----- TODO -----

    pass

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''

    train_data = np.load("../data/train_data.npz")
    # ----- TODO -----

    pass
