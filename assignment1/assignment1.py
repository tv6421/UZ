# COMMON ERRORS AND DEBUG IDEAS
# • check your x and y coordinates
# • check your data type: float, uint8
# • check your data range: [0,255], [0,1]
# • perform simple checks (synthetic data examples)

from UZ_utils import *
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
################################################
# Exercise 1: Basic image processing

##### a) #####
def naloga_1a():
    I = imread('images/umbrellas.jpg')
    print(I.dtype)        # == float64
    imshow(I)
    I = cv2.imread('images/umbrellas.jpg')[0] # BGR
    I = plt.imread('images/umbrellas.jpg')[0] # RGB
    I_float = I.astype(np.float64)
    print(I_float)

##### b) #####
def naloga_1b():
    I = cv2.imread('images/umbrellas.jpg') # BGR
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB) # changes to RGB
    I_float = I.astype(np.float64)
    I_gray = (I_float[:,:,0] + I_float[:,:,1] + I_float[:,:,2]) / 3
    imshow(I_gray)

##### c) #####
def naloga_1c():
    I = cv2.imread('images/umbrellas.jpg') # BGR
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB) # changes BGR to RGB
    I_float = I.astype(np.float64)
    I_gray = (I_float[:,:,0] + I_float[:,:,1] + I_float[:,:,2]) / 3
    cutout = I[130:260, 240:450, 1]
    plt.subplot(1, 2, 1)
    plt.imshow(I_gray, cmap="gray")
    plt.title("GRAY")

    plt.subplot(1, 2, 2)
    plt.imshow(cutout, cmap="gray")
    plt.title("CUTOUT")
    plt.show()

##### d) #####
def naloga_1d():
    I = imread('images/umbrellas.jpg')
    I_gray = (I[:,:,0] + I[:,:,1] + I[:,:,2]) / 3
    for i in range(130, 260):
        for j in range(240, 450):
            I[i,j] = 1 - I[i, j]
            # print(I[i,j])
    imshow(I)

##### e) #####
def naloga_1e():
    I = imread_gray('images/umbrellas.jpg')
    I_rescaled = I * 0.3
    plt.subplot(1, 2, 1)
    plt.imshow(I, vmin=0, vmax=1, cmap="gray")
    plt.title("I")

    plt.subplot(1, 2, 2)
    plt.imshow(I_rescaled, vmin=0, vmax=1, cmap="gray")
    plt.title("I_rescaled")
    plt.show()

################################################
# Exercise 2: Thresholding and histograms

##### a) #####
def naloga_2a():
    I = imread_gray('images/bird.jpg')
    threshold = 0.3
    J = np.where(I < threshold, 0, 1)

    plt.subplot(1, 2, 1)
    plt.imshow(I, vmin=0, vmax=1, cmap="gray")
    plt.title("img")

    plt.subplot(1, 2, 2)
    plt.imshow(J, vmin=0, vmax=1, cmap="gray")
    plt.title("mask")
    plt.show()

##### b) #####
def naloga_2b():
    I = imread_gray('images/bird.jpg')  
    plt.subplot(2,2,1)
    plt.imshow(I, cmap='gray')
    plt.subplot(2,2,2)
    plt.bar(range(100), myhist(I, 100))
    plt.subplot(2,2,3)
    plt.bar(range(20), myhist(I, 20))
    plt.subplot(2,2,4)
    plt.bar(range(255), myhist(I, 255))
    plt.show()

def myhist(I, num_bins):
    H = np.zeros(num_bins)
    I = I.reshape(-1)
    value_range = 1 / num_bins
    for i in I:
        ix = math.floor(i * num_bins)
        if ix < num_bins: H[ix] += 1
        else: H[num_bins - 1] += 1
    return H / np.sum(H) # or H/len(I)

##### c) #####
def naloga_2c():
    I = imread_gray('images/bird.jpg') 
    plt.subplot(3,2,1)
    plt.bar(range(100), myhist(I, 100))
    plt.title("myhist, myhist2, diff (100)")
    plt.subplot(3,2,2)
    plt.bar(range(20), myhist(I, 20))
    plt.title("myhist, myhist2, diff (20)")

    plt.subplot(3,2,3)
    plt.bar(range(100), myhist2(I, 100))
    plt.subplot(3,2,4)
    plt.bar(range(20), myhist2(I, 20))
    
    plt.subplot(3,2,5)
    plt.bar(range(100), myhist2(I, 100) - myhist(I, 100))
    plt.subplot(3,2,6)
    plt.bar(range(20), myhist2(I, 20)- myhist(I, 20))
    plt.show()

def myhist2(I, num_bins):
    H = np.zeros(num_bins)
    I = I.reshape(-1)
    min_val = np.min(I)
    max_val = np.max(I)
    value_range = (max_val - min_val) / num_bins
    for i in I:
        ix = int((i - min_val) // value_range)    
        if ix < num_bins: H[ix] += 1
        else: H[num_bins - 1] += 1
    return H / np.sum(H) # or H/len(I)
    
##### d) #####
def naloga_2d():
    I1 = imread_gray('images/bird1.jpg') 
    I2 = imread_gray('images/bird2.jpg') 
    I3 = imread_gray('images/bird3.jpg')
    plt.subplot(3,3,1)
    plt.bar(range(20), myhist(I1, 20))
    plt.title("high lighting")
    plt.subplot(3,3,2)
    plt.bar(range(20), myhist(I2, 20))
    plt.title("neutral lighting")
    plt.subplot(3,3,3)
    plt.bar(range(20), myhist(I3, 20))
    plt.title("low lighting")

    plt.subplot(3,3,4)
    plt.bar(range(100), myhist(I1, 100))
    plt.subplot(3,3,5)
    plt.bar(range(100), myhist(I2, 100))
    plt.subplot(3,3,6)
    plt.bar(range(100), myhist(I3, 100))
    
    plt.subplot(3,3,7)
    plt.bar(range(255), myhist(I1, 255))
    plt.subplot(3,3,8)
    plt.bar(range(255), myhist(I2, 255))
    plt.subplot(3,3,9)
    plt.bar(range(255), myhist(I3, 255))
    plt.show()

##### e) #####
def naloga_2e():
    otsu(imread_gray('images/bird.jpg'))

def otsu(im):
    im = np.uint8(im * 255)
    thresholds = []
    for threshold in range(0, 256):
        thresholded_im = np.zeros(im.shape)
        thresholded_im[im >= threshold] = 1

        nb_pixels = im.size
        nb_pixels1 = np.count_nonzero(thresholded_im)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1

        if weight1 == 0 or weight0 == 0: continue

        val_pixels1 = im[thresholded_im == 1]
        val_pixels0 = im[thresholded_im == 0]

        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
        thresholds.append(weight0 * var0 + weight1 * var1)

    best_threshold = np.argmin(thresholds)
    # print(best_threshold / 256)
    return best_threshold / 256
    
################################################
# Exercise 3: Morphological operations and regions

##### a) #####
def naloga_3a():
    I = imread_gray('images/bird.jpg')
    n = 5
    SE = np.ones((n,n)) # create a square structuring element
    I_closed5 = cv2.dilate(I, SE)
    I_closed5 = cv2.erode(I_closed5, SE)
    I_opened5 = cv2.erode(I, SE)
    I_opened5 = cv2.dilate(I_opened5, SE)
    n = 3
    SE = np.ones((n,n)) # create a square structuring element
    I_closed3 = cv2.dilate(I, SE)
    I_closed3 = cv2.erode(I_closed3, SE)
    I_opened3 = cv2.erode(I, SE)
    I_opened3 = cv2.dilate(I_opened3, SE)
    n = 10
    SE = np.ones((n,n)) # create a square structuring element
    I_closed10 = cv2.dilate(I, SE)
    I_closed10 = cv2.erode(I_closed10, SE)
    I_opened10 = cv2.erode(I, SE)
    I_opened10 = cv2.dilate(I_opened10, SE)

    plt.subplot(3,2,1)
    plt.imshow(I_closed5, cmap='gray')
    plt.title("closing (n = 5, 3, 10)")
    plt.subplot(3,2,2)
    plt.imshow(I_opened5, cmap='gray')
    plt.title("opening (n = 5, 3, 10)")

    plt.subplot(3,2,3)
    plt.imshow(I_closed3, cmap='gray')
    # plt.title("myhist2(I, 100)")
    plt.subplot(3,2,4)
    plt.imshow(I_opened3, cmap='gray')
    # plt.title("myhist2(I, 20)")
    
    plt.subplot(3,2,5)
    plt.imshow(I_closed10, cmap='gray')
    plt.subplot(3,2,6)
    plt.imshow(I_opened10, cmap='gray')
    plt.show()

##### b) #####
def naloga_3b():
    I = imread_gray('images/bird.jpg')
    imshow(create_mask(I, 0.21))

def create_mask(I, threshold = None):
    # threshold = 0.21 # 0.7  #
    if threshold == None:
        threshold = otsu(I)
    I = np.where(I < threshold, 0, 1)
    I = I.astype(np.uint8)
    n = 15 #; SE = np.ones((n,n))
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    I = cv2.erode(cv2.dilate(I, SE), SE)
    return I

##### c) #####
def naloga_3c():
    I = imread('images/bird.jpg')
    J = imread_gray('images/bird.jpg')
    masked = immask(I, create_mask(J, 0.21))
    imshow(masked)

def immask(I, mask):
    if len(I.shape) > 2 and len(mask.shape) == 2:
        mask = np.expand_dims(mask, 2)
    I = np.where(mask == 0, 0, I)
    return I

##### d) #####
def naloga_3d():
    I = imread('images/eagle.jpg')
    J = imread_gray('images/eagle.jpg')
    threshold = 0.71
    J[J < threshold] = 0
    J[J >= threshold] = 1
    J = J.astype(np.uint8)
    n = 10
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    J = cv2.erode(J, SE)
    J = cv2.dilate(J, SE)
    imshow(immask(I, J))
    J = np.where(J > threshold, 0, 1)
    imshow(J)

##### e) #####
def naloga_3e():
    I = imread_gray('images/coins.jpg')
    I = np.where(I < 0.99, 1, 0)
    I = np.asarray(I, np.uint8)
    n = 15
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    I = cv2.dilate(I, SE)
    I = cv2.erode(I, SE)
    I = cv2.erode(I, SE)
    # imshow(I)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(I, connectivity=8)
    result = np.copy(I)
    for i, (_, _, _, _, area) in enumerate(stats):
        if area > 700:
            result[labels == i] = 0
    imshow(result)

naloga_2a()
naloga_2b()
naloga_2c()
naloga_2d()
naloga_2e()
naloga_3a()
naloga_3b()
naloga_3c()
naloga_3d()
naloga_3e()