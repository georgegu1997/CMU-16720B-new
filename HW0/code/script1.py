import numpy as np
from alignChannels import alignChannels, alignTwoNCC, alignTwoSSD

import matplotlib.pyplot as plt

# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load("../data/red.npy")
green = np.load("../data/green.npy")
blue = np.load("../data/blue.npy")

'''
Actually two metrics yield almost the same output
'''
# 2. Find best alignment
rgbResultSSD = alignChannels(red, green, blue, alignTwoSSD)
rgbResultNCC = alignChannels(red, green, blue, alignTwoNCC)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
plt.imsave("../results/rgbResultSSD.jpg", rgbResultSSD)
plt.imsave("../results/rgbResultNCC.jpg", rgbResultNCC)
plt.imsave("../results/rgb_output.jpg", rgbResultNCC)
