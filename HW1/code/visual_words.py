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
TEMP_ROOT_SAMPLED_RESPONSE = "../temp/response/"

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
    if image.dtype == np.uint8 or image.max() > 10:
        image = image.astype(float) / 255.0
    # Check and convert the dimensions
    if image.ndim == 2:
        image = np.stack([image]*3, axis=2)

    # Convert the image to Lab color space
    image = skimage.color.rgb2lab(image)

    H, W, _ = image.shape
    filter_responses = []

    for s in FILTER_SCALES: # iterate over scales
        for c in range(3):
            # Gaussian
            filter_responses.append(scipy.ndimage.gaussian_filter(image[:,:,c], sigma=s))
        for c in range(3):
            # Laplacian of Gaussian
            filter_responses.append(scipy.ndimage.gaussian_laplace(image[:,:,c], sigma=s))
        for c in range(3):
            # x derivative of Gaussian
            filter_responses.append(scipy.ndimage.gaussian_filter(image[:,:,c], sigma=s, order=[0, 1]))
        for c in range(3):
            # y derivative of Gaussian
            filter_responses.append(scipy.ndimage.gaussian_filter(image[:,:,c], sigma=s, order=[1, 0]))


    filter_responses = np.stack(filter_responses, axis=2)
    return filter_responses

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * dictionary: numpy.ndarray of shape (K, 3F)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    # extract the filter responses
    filter_responses = extract_filter_responses(image) # Will be (H, W, 3F)
    H, W, _ = filter_responses.shape
    filter_responses = filter_responses.reshape(H*W, -1)

    # Compute the distance and assign label
    visual_words = scipy.spatial.distance.cdist(filter_responses, dictionary, metric='euclidean') # Will be (H*W, K)
    visual_words = np.argmin(visual_words, axis = 1)
    visual_words = visual_words.reshape((H, W))

    return visual_words


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
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255
    filter_responses = extract_filter_responses(image)
    H, W, _ = image.shape

    # Flatten the pixels for sampling.
    filter_responses = filter_responses.reshape([H*W, -1])
    sample_indices = np.random.choice(H*W, alpha, replace=False)
    sampled_response = filter_responses[sample_indices]

    print("processed image %d" % i)

    # Save the sampled_response
    np.save(os.path.join(TEMP_ROOT_SAMPLED_RESPONSE, "%d"%i), sampled_response)

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
    ALPHA = 275
    K = 200
    image_paths = train_data['files']
    image_labels = train_data['labels']

    # Construct the batches for multiprocessing pool
    batches = []
    for i, p in enumerate(image_paths):
        image_path = os.path.join("../data/", p)
        batches.append((i, ALPHA, image_path))

    # multiprocessing on compute_dictionary_one_image
    pool = multiprocessing.Pool(num_workers)
    pool.map(compute_dictionary_one_image, batches)
    print("compute_dictionary: All the responses are saved")

    # Read the responses and perform the K-Means algorithm
    response_paths = [os.path.join(TEMP_ROOT_SAMPLED_RESPONSE, f) \
        for f in os.listdir(TEMP_ROOT_SAMPLED_RESPONSE) \
        if os.path.isfile(os.path.join(TEMP_ROOT_SAMPLED_RESPONSE, f))]

    # concatenate it to a large array for training
    responses = []
    for response_path in response_paths:
        responses.append(np.load(response_path))
    responses = np.concatenate(responses, axis = 0)
    print("responses.shape:", responses.shape)

    # Run the K-Means algorithm
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=-1).fit(responses)
    dictionary = kmeans.cluster_centers_
    print("dictionary.shape:", dictionary.shape)

    np.save("dictionary", dictionary)
