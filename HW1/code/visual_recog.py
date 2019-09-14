import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    SPM_layer_num = 3
    K = dictionary.shape[0]
    image_paths = train_data['files']
    image_labels = train_data['labels']

    # Construct the batches for multiprocessing pool
    batches = []
    for i, p in enumerate(image_paths):
        image_path = os.path.join("../data/", p)
        batches.append((image_path, dictionary, SPM_layer_num, K))

    # Extract the image feature using multiprocessing
    pool = multiprocessing.Pool(num_workers)
    print("Multiprocessing start for get_image_feature()...", end=" ")
    features = pool.starmap(get_image_feature, batches)
    print("Done")

    features = np.stack(features, axis = 0)
    print("features.shape:", features.shape)

    # Save the results
    np.savez("trained_system.npz", features=features, labels=image_labels, dictionary=dictionary, SPM_layer_num=SPM_layer_num)

def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----

    pass

def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----
    # Read the image
    image = skimage.io.imread(file_path)
    image = image.astype('float')/255

    # Get the Get the SPM feature and return it
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)

    return feature


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    N, K  = histograms.shape
    minimum = np.minimum(word_hist.reshape((1, K)), histograms)
    sim = minimum.sum(axis = 1)

    return sim


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    word = wordmap.reshape(-1)

    # the values in the word will be from 0 to K-1
    hist, _ = np.histogram(word, bins=np.arange(dict_size+1)-0.5, density=True)

    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers L+1
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----
    hist_all = []
    L = layer_num - 1
    K = dict_size

    # The sizes are from 1/1 to 1/(2^L)
    # split the image to grid on the finest level
    H, W = wordmap.shape
    finest_shape = (int(H / (2**L)), int(W / (2**L)))
    finest_image_grid = skimage.util.view_as_windows(wordmap, finest_shape, step=finest_shape)
    # print(finest_image_grid.shape) # (4, 4, 93, 125)

    # Compute the histgram on the finest level
    finest_image_list = finest_image_grid.reshape(( (2**L)*(2**L) , *finest_shape))
    finest_hist_list = []
    for w in finest_image_list:
        finest_hist_list.append(get_feature_from_wordmap(w, dict_size))
    finest_hist_list = np.stack(finest_hist_list, axis = 0)
    finest_hist_grid = finest_hist_list.reshape(((2**L), (2**L), dict_size))

    # Compute, normalize and append the histgram for each level
    for l in range(L, -1, -1):
        s = int((2**L) / (2**l)) # size of this level in term of finest_window
        hist_grid = skimage.util.view_as_windows(finest_hist_grid, (s, s, dict_size), step = (s, s, dict_size))
        # eliminate the extra dimension
        hist_grid = np.squeeze(hist_grid, axis = 2)
        # Sum up the histograms that are in a region of several finest grid
        hist_grid = hist_grid.sum(axis = (2, 3))
        # normalize histograms in all window in this level (Then they sum up to 1)
        hist_grid = hist_grid / (4**L)

        # Flatten and L1 normalize
        if l == 0 or l == 1:
            normalize_factor = 2**(-L)
        else:
            normalize_factor = (2**(l-L-1))
        hist_list = hist_grid.reshape((-1, dict_size)) * normalize_factor
        hist_all.append(hist_list)

    # concatenate all the features and Flatten them
    hist_all = np.concatenate(hist_all, axis = 0)
    hist_all = hist_all.reshape(-1)
    # print(hist_all.shape)
    # print(K * (4**(L+1)-1) / 3)
    # print(hist_all.sum())
    return hist_all
