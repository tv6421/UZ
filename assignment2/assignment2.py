# COMMON ERRORS AND DEBUG IDEAS
# • check your x and y coordinates
# • check your data type: float, uint8
# • check your data range: [0,255], [0,1]
# • perform simple checks (synthetic data examples)

from a2_utils import *
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
from PIL import Image


################################################
def imread(path):
    I = Image.open(path).convert('RGB')  # PIL image.
    I = np.asarray(I)  # Converting to Numpy array.
    I = I.astype(np.float64) / 255
    return I
def imread_gray(path):
    I = Image.open(path).convert('L')  # PIL image opening and converting to gray.
    I = np.asarray(I)  # Converting to Numpy array.
    I = I.astype(np.float64) / 255
    return I
def image_read(path):
    I = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
################################################
# Exercise 1: Convolution

##### 1b) #####
def naloga_1b():
    signal = read_data("signal.txt")
    kernel = read_data("kernel.txt")

    plt.plot(cv2.filter2D(signal, -1, kernel), "red")
    plt.plot(signal, "blue")
    plt.plot(kernel, "orange")
    plt.plot(simple_convolution(signal, kernel), "green")
    plt.show()

def simple_convolution(I, k):
    # k = 2 * N + 1 
    N = int((len(k) - 1) / 2)  # print(N)
    res = []
    for i in range(N, len(I) - N): #- 1):
        sum_ = 0
        for j in range(len(k)):
            sum_ += k[j] * I[i - j]
        res.append(sum_)
    # print(res)
    return res


##### 1c) #####
def naloga_1c():
    signal = read_data("signal.txt")
    kernel = read_data("kernel.txt")

    plt.plot(cv2.filter2D(signal, -1, kernel), "red")
    plt.plot(signal, "blue")
    plt.plot(kernel, "orange")
    plt.plot(not_simple_convolution(signal, kernel), "green")
    plt.show()

def not_simple_convolution(I, k):
    N = int((len(k) - 1) / 2)  # print(N)
    # res = np.zeros(len(I))
    res = []
    for i in range(N, len(I) - N): #- 1):
        sum_ = 0
        for j in range(len(k)):
            if (i - N + j) >= 0 and (i - N + j) < len(I): 
                sum_ += k[j] * I[i - j]
        # res[i] = sum_
        res.append(sum_)
    return res
    # return cv2.filter2D(I, -1, np.array(k))
    

##### 1d) #####
def naloga_1d():
    # kernel_size = np.ceil(2 * sigma) + 1 == 2 * 2 + 1 = 5
    plt.plot(get_range(0.5), gauss(0.5), "blue", label = "sigma = 0.5")
    plt.plot(get_range(1), gauss( 1), "orange", label = "sigma = 1")
    plt.plot(get_range(2), gauss(2), "green", label = "sigma = 2")
    plt.plot(get_range(3), gauss(3), "red", label = "sigma = 3")
    plt.plot(get_range(4), gauss(4), "purple", label = "sigma = 4")
    plt.legend()
    plt.show()

# calculates a Gaussian kernel
def gauss(sigma):
    x = get_range(sigma)
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * \
            np.exp(-(x * x) / (2 * sigma * sigma))

def get_range(sigma):
    x = 2 * np.ceil(3 * sigma) + 1
    start = -(x // 2)
    end = start + x
    step = 1
    return np.arange(start, end, step)

##### 1e) #####
def naloga_1e(): 
    I = read_data("signal.txt")
    k1 = gauss(2)
    k2 = [0.1, 0.6, 0.4]

    plt.figure(figsize = (15, 5))
    plt.subplot(1, 4, 1)
    plt.title("s")
    plt.plot(I)

    plt.subplot(1, 4, 2)
    plt.title("(s * k1) * k2")
    convolved_k1 = not_simple_convolution(I, k1)
    convolved_k1_k2 = not_simple_convolution(convolved_k1, k2)
    plt.plot(convolved_k1_k2)
    
    plt.subplot(1, 4, 3)
    plt.title("(s * k2) * k1")
    convolved_k2 = not_simple_convolution(I, k2)
    convolved_k2_k1 = not_simple_convolution(convolved_k2, k1)
    plt.plot(convolved_k2_k1)

    plt.subplot(1, 4, 4)
    plt.title("s * (k1 * k2)")
    k3 = not_simple_convolution(k1, k2)
    convolved_k3 = not_simple_convolution(I, k3)
    plt.plot(convolved_k3)

    plt.show()



################################################
# Exercise 2: Image filtering

##### 2a) #####
def naloga_2a():
    I = imread_gray("images/lena.png")

    gaussian_noise = gauss_noise(I)
    salt_pepper_noise = sp_noise(I)
    filtered_gaussian_noise = gaussfilter(gaussian_noise, 1)
    filtered_salt_pepper = salt_pepper_noise

    plt.figure(figsize = (12, 7))
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(I, cmap="gray")
    plt.subplot(2, 3, 2)
    plt.title("Gaussian noise")
    plt.imshow(gaussian_noise, cmap="gray")
    plt.subplot(2, 3, 3)
    plt.title("Salt and peper")
    plt.imshow(salt_pepper_noise, cmap="gray")
    plt.subplot(2, 3, 5)
    plt.title("Filtered Gaussian noise")
    plt.imshow(filtered_gaussian_noise, cmap="gray")
    plt.subplot(2, 3, 6)
    plt.title("Filtered salt and pepper")
    plt.imshow(filtered_salt_pepper, cmap="gray")
    plt.show()

def gaussfilter(I, sigma):   # cv2.filter2D(src, ddepth, kernel)
    gaussian_kernel = gauss(sigma)
    I = cv2.filter2D(I, -1, gaussian_kernel)
    I = I.T
    I = cv2.filter2D(I, -1, gaussian_kernel)
    return I.T
    # gaussian_kernel = gauss(sigma)
    # return cv2.filter2D(cv2.filter2D(I, -1, gaussian_kernel).T, -1, gaussian_kernel).T

##### 2b) #####
def naloga_2b():
    I = cv2.cvtColor(cv2.imread("images/museum.jpg"), cv2.COLOR_BGR2GRAY)
    I = np.array(I)
    # https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    k = np.array([ [0, -1, 0 ],   
                   [-1, 5, -1],
                   [0, -1, 0 ] ])

    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.filter2D(I, -1, k), cmap="gray")
    plt.title("Sharpened")
    plt.show()

##### 2c) #####
def naloga_2c():
    # signal = np.zeros(40)
    # signal[10:20] = 1
    signal = read_data("signal.txt")
    corrupted = np.copy(signal)
    ix_salt = np.random.choice(len(signal), size = 3, replace = False)
    ix_pepper = np.random.choice(len(signal), size = 3, replace = False)
    corrupted[ix_salt] = np.max(signal)
    corrupted[ix_pepper] = np.min(signal)

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.plot(signal)

    plt.subplot(1, 4, 2)
    plt.title("Corrupted")
    plt.plot(corrupted)
    
    plt.subplot(1, 4, 3)
    plt.title("Gauss")
    plt.plot(gaussfilter(corrupted, 1))
    
    plt.subplot(1, 4, 4)
    plt.title("Median")
    plt.plot(simple_median(corrupted, 3))
    
    plt.show()

def simple_median(I, w):
    I = I.flatten()
    res = np.copy(I)
    offset = int(w / 2)
    for i in range(offset, len(I) - offset):
        res[i] = np.median(I[i - offset:i + offset])
    return res

##### 2d) #####
def naloga_2d():
    I = imread_gray("images/lena.png")

    plt.figure(figsize = (20, 7))
    plt.subplot(2, 4, 1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")

    plt.subplot(2, 4, 2)
    plt.imshow(gauss_noise(I), cmap="gray")
    plt.title("Gausian noise")

    plt.subplot(2, 4, 3)
    plt.imshow(gaussfilter(gauss_noise(I), 1), cmap="gray")
    plt.title("Gauss filtered GN")

    plt.subplot(2, 4, 4)
    plt.imshow(not_simple_median(gauss_noise(I), 3), cmap="gray")
    plt.title("Median filtered GN")

    plt.subplot(2, 4, 6)
    plt.imshow(sp_noise(I), cmap="gray")
    plt.title("Salt and pepper")

    plt.subplot(2, 4, 7)
    plt.imshow(gaussfilter(sp_noise(I), 1), cmap="gray")
    plt.title("Gauss filtered SP")
    
    plt.subplot(2, 4, 8)
    plt.imshow(not_simple_median(gauss_noise(I), 3), cmap="gray")
    plt.title("Median filtered SP")

    plt.show()

def not_simple_median(I, w):
    res = np.zeros(I.shape)
    for i in range(len(I) - w):
        for j in range(len(I[0]) - w):
            patch = np.array(I[i:i + w, j:j+w])
            res[i + int(w / 2), j + int(w / 2)] = np.median(patch)
    return res

################################################
# Exercise 3: Global approach to image description

##### 3a) #####
def myhist3(I, n_bins):
    # the function should also work on other color space, if I != RGB: convert
    H = np.zeros((n_bins, n_bins, n_bins))
    # print(np.shape(I))
    for row in I:
        for pixel in row:
            # print(pixel[0], pixel[1], pixel[2])
            R, G, B = pixel[0], pixel[1], pixel[2]
            H[int(R * n_bins), int(G * n_bins), int(B * n_bins)] += 1
    # H = H.flatten()
    return H / np.sum(H)

##### 3b) #####
def compare_histograms(hist1, hist2, measure = None):
    # if size(hist1) != size(hist1): print("histogram bins different sizes")

    h1 = hist1.flatten() #hist1.resize(-1)
    h2 = hist2.flatten() #hist2.resize(-1)

    # euclidean_distance()
    L2 = np.sqrt(np.sum((h1 - h2)**2))
    if measure == "L2":
       return L2

    # chi_square():
    e0 = 1e-10
    chi = np.sum(((h1 - h2) ** 2) / (h1 + h2 + e0)) / 2
    if measure == "chi":
        return chi

    # intersection()
    intersection = 1 - np.sum(np.minimum(h1, h2))
    if measure == "inter":
        return intersection

    # hellinger_distance()
    hellinger = np.sqrt(np.sum((np.sqrt(h1) - np.sqrt(h2)) ** 2) / 2)
    if measure == "hell":
        return hellinger
    
    return (L2, chi, intersection, hellinger)

##### 3c) #####
def naloga_3c():
    # I1 = plt.imread("dataset/object_01_1.png")
    # I1 = cv2.cvtColor(cv2.imread("dataset/object_01_1.png"), cv2.COLOR_BGR2RGB)
    I1 = imread("dataset/object_01_1.png")
    I2 = imread("dataset/object_02_1.png")
    I3 = imread("dataset/object_03_1.png")

    h1 = myhist3(I1, 8)
    h2 = myhist3(I2, 8)
    h3 = myhist3(I3, 8)

    plt.subplot(2, 3, 1)
    plt.imshow(I1)
    plt.subplot(2, 3, 2)
    plt.imshow(I2)
    plt.subplot(2, 3, 3)
    plt.imshow(I3)
    plt.subplot(2, 3, 4)
    plt.bar(range(512), h1.flatten())
    plt.subplot(2, 3, 5)
    plt.bar(range(512), h2.flatten())
    plt.subplot(2, 3, 6)
    plt.bar(range(512), h3.flatten())
    plt.show()

    print(f"L2 between 1 and 2: {compare_histograms(h1, h2, 'L2')}")
    print(f"L2 between 1 and 3: {compare_histograms(h1, h3, 'L2')}")

##### 3d) #####
class image_data():
    def __init__(self, name, I, hist): #, L2, chi, inter, hell):
        self.name = name
        self.I = I
        self.hist = hist
        self.L2 = 0
        self.chi = 0
        self.inter = 0
        self.hell = 0

def naloga_3d(n_bins = 8):
    ref_img = imread("dataset/object_05_4.png")
    ref_hist = myhist3(ref_img, n_bins)
    path_hists = get_path_histograms("dataset", n_bins)
    
    for curr in path_hists:
        curr.L2, curr.chi, curr.inter, curr.hell = compare_histograms(ref_hist, curr.hist)
        # print(curr.L2)

    sorted_by_hell = sorted(path_hists, key = lambda x: x.hell)[0:6]
    plt.figure(figsize = (20, 7))
    for ix, obj in enumerate(sorted_by_hell):
        plt.subplot(2, 6, ix + 1)
        plt.imshow(obj.I)
        plt.subplot(2, 6, ix + 7)
        plt.bar(range(n_bins * n_bins * n_bins), obj.hist)
        plt.title(f"hell = {str(round(obj.hell, 2))}")
    plt.show()

    # sorted_by_L2 = sorted(path_hists, key = lambda x: x.L2)
    # for ix, obj in enumerate(sorted_by_L2):
    #     plt.subplot(2, 4, ix + 1)
    #     plt.imshow(obj.I)
    #     plt.subplot(2, 4, ix + 5)
    #     plt.bar(range(n_bins * n_bins * n_bins), obj.hist)
    #     plt.title(f"L2 = {str(round(obj.L2, 2))}")
    # plt.show()

    # sorted_by_chi = sorted(path_hists, key = lambda x: x.chi)
    # for ix, obj in enumerate(sorted_by_chi):
    #     plt.subplot(2, 4, ix + 1)
    #     plt.imshow(obj.I)
    #     plt.subplot(2, 4, ix + 5)
    #     plt.bar(range(n_bins * n_bins * n_bins), obj.hist)
    #     plt.title(f"chi = {str(round(obj.chi, 2))}")
    # plt.show()

    # sorted_by_inter = sorted(path_hists, key = lambda x: x.inter)
    # for ix, obj in enumerate(sorted_by_inter):
    #     plt.subplot(2, 4, ix + 1)
    #     plt.imshow(obj.I)
    #     plt.subplot(2, 4, ix + 5)
    #     plt.bar(range(n_bins * n_bins * n_bins), obj.hist)
    #     plt.title(f"inter = {str(round(obj.inter, 2))}")
    # plt.show()

# retrieval_system
def get_path_histograms(path, n_bins):
    hists = []
    for img_name in os.listdir(path): #[16:21]:
        name = f"dataset/{img_name}"
        I = imread(name)
        hist = myhist3(I, n_bins)
        im = image_data(name, I, hist.flatten())
    #     name_img_hist = (I_name, I, hist.flatten())
        hists.append(im)
    return hists

##### 3e) #####
def naloga_3e(n_bins = 8):
    ref_img = imread("dataset/object_05_4.png")
    ref_hist = myhist3(ref_img, n_bins)
    path_hists = get_path_histograms("dataset", n_bins)
    
    distances = []
    for curr in path_hists:
        distances.append(compare_histograms(ref_hist, curr.hist, "hell"))
    
    image_indices = np.arange(len(distances))
    sorted_distances = sorted(distances)#[0:5]
    similar_indices = [ix for ix, dist in enumerate(distances) if dist in sorted_distances[0:5]]

    # unsorted distances
    plt.figure(figsize = (10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(image_indices, distances)
    plt.scatter(similar_indices, [dist for dist in distances if dist in sorted_distances[0:5]], marker="o", c="red")
    plt.title("Unsorted distances")

    # sorted distances
    plt.subplot(1, 2, 2)
    plt.plot(image_indices, sorted_distances)
    plt.scatter(range(5), [dist for dist in sorted_distances[0:5]], marker="o", c="red")
    plt.title("Sorted distances")
    plt.show()

##### 3f) #####
def naloga_3f(n_bins = 8):
    hists = get_path_histograms("dataset", n_bins)
    res = np.sum(np.array([H.hist for H in hists]), axis = 0) / len(hists)

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111, projection='3d')
    
    xyz = np.array([[i, j, k] for i in range(n_bins) for j in range(n_bins) for k in range(n_bins)])
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c = xyz / n_bins, s = res * 1000, alpha = 1)    # c for colors, s for size, alpha for opacity
    ax1.set_xlabel("red")
    ax1.set_ylabel("green")
    ax1.set_zlabel("blue")
    plt.show()

naloga_3f()