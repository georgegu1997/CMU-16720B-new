import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers

class ReducedVGG16():
    """docstring for ReducedVGG16."""
    def __init__(self, vgg16):
        self.vgg16 = vgg16
        self.vgg16.eval()
        # Truncate the classifier network to the second linear layer
        self.net_classifier = torch.nn.Sequential(*list(list(vgg16.children())[2])[:4])
        self.net_classifier.eval()

    def forward(self, x):
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net_classifier(x)
        return x

def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")

    # ----- TODO -----
    image_paths = train_data['files']
    image_labels = train_data['labels']

    # Construct the batches for multiprocessing pool
    batches = []
    for i, p in enumerate(image_paths):
        image_path = os.path.join("../data/", p)
        batches.append((i, image_path, vgg16))

    # Extract the image feature using multiprocessing
    pool = multiprocessing.Pool(num_workers)
    print("Multiprocessing start for get_image_feature()...", end=" ")
    features = pool.map(get_image_feature, batches)
    print("Done")

    features = np.stack(features, axis = 0)
    print("features.shape:", features.shape)

    # Save the results
    np.savez("trained_system_deep.npz", features=features, labels=image_labels)

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system_deep.npz")

    # ----- TODO -----
    NUM_CLASSES = 8
    test_paths = test_data['files']
    test_labels = test_data['labels']
    train_features = trained_system['features']
    train_labels = trained_system['labels']

    # Construct the batches for multiprocessing pool
    batches = []
    for i, p in enumerate(test_paths):
        image_path = os.path.join("../data/", p)
        batches.append((i, image_path, vgg16))

    # Extract the image feature using multiprocessing
    pool = multiprocessing.Pool(num_workers)
    print("Multiprocessing start for get_image_feature()...", end=" ")
    test_features = pool.map(get_image_feature, batches)
    print("Done")

    # Construct the confusion matrix
    conf = np.zeros((NUM_CLASSES,NUM_CLASSES))
    for x, y in zip(test_features, test_labels):
        similarity = distance_to_set(x, train_features)
        pred_y = train_labels[similarity.argmax()]
        conf[y, pred_y] += 1
    # Compute the accuracy
    accuracy = np.diag(conf).sum()/conf.sum()

    return conf, accuracy

def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''
    # ----- TODO -----
    # Only for dtype conversion and value normalization
    image = network_layers.preprocess_image(image)
    # additional step on input for torch inference
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    return image


def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.

    [output]
    * feat: evaluated deep feature
    '''

    i, image_path, vgg16 = args

    # ----- TODO -----
    # Load the image
    image = skimage.io.imread(image_path)
    image = image.astype('float')/255
    x = preprocess_image(image) # Already tensor
    x = x.reshape((1, *x.shape))

    net = ReducedVGG16(vgg16)
    y = net.forward(x)
    feat = y.detach().numpy().squeeze()

    return feat


def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    feature = feature.reshape((1, -1))
    dist = ((train_features - feature)**2).sum(axis=1)
    dist = - dist**(1/2)
    return dist
