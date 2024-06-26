# COMMON ERRORS AND DEBUG IDEAS
# • check your x and y coordinates
# • check your data type: float, uint8
# • check your data range: [0,255], [0,1]
# • perform simple checks (synthetic data examples)

from a3_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os


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

def imshow(img, title=None):
    if len(img.shape) == 3:
        plt.imshow(img)  # if type of data is "float", then values have to be in [0, 1]
    else:
        plt.imshow(img)
        plt.set_cmap('gray')  # also "hot", "nipy_spectral"
        plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
################################################
# Exercise 1: Image derivatives

##### 1a) #####
# compute the first and second derivative with respect to y: 
# Iy(x, y), Iyy(x, y), as well as the mixed derivative Ixy(x, y).
#   Solutions:
# Iy(x; y) = d/dx [g(x) * g(y) * I(x,y)] = d/dx g(y) * [g(x) * I(x,y)]
# Iyy(x; y) = d/dx [g(x) * g(y) * Iy(x,y)] = d/dx g(y) * [g(x) * Iy(x,y)]
# Ixy(x; y) = d/dx [g(x) * g(y) * Ix(x,y)] = d/dx g(y) * [g(x) * Ix(x,y)]

##### 1b) #####
def naloga_1b():
    pass

def gauss(sigma):
    x = get_range(sigma)
    g = (1 / (np.sqrt(2 * np.pi) * sigma)) * \
            np.exp(-(x * x) / (2 * sigma * sigma))
    return g / np.sum(g)

def gaussdx(sigma):
    x = get_range(sigma)
    g = -(x * np.exp(-((x ** 2) / (2 * sigma ** 2))) \
        / (np.sqrt(2 * np.pi) * (sigma ** 3)))
    # print(np.sum(np.abs(g)))
    return g / np.sum(np.abs(g))

def get_range(sigma):
    x = 2 * np.ceil(3 * sigma) + 1
    start = -(x // 2)
    end = start + x
    step = 1
    return np.arange(start, end, step)

##### 1c) #####
def naloga_1c():
    G = gauss(2)
    D = gaussdx(2)

    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1
    plt.set_cmap("gray")
    plt.subplot(2,3,1)
    plt.imshow(impulse)
    plt.title("Impulse")

    I = cv2.filter2D(impulse, -1, G)
    I = cv2.filter2D(I.T, -1, D)
    plt.subplot(2,3,2)
    plt.imshow(I)
    plt.title("G, Dt")

    I = cv2.filter2D(impulse, -1, D)
    I = cv2.filter2D(I.T, -1, G)
    plt.subplot(2,3,3)
    plt.imshow(I)
    plt.title("D, Gt")

    I = cv2.filter2D(impulse, -1, G)
    I = cv2.filter2D(I.T, -1, G)
    plt.subplot(2,3,4)
    plt.imshow(I)
    plt.title("G, Gt")

    I = cv2.filter2D(impulse, -1, G.T)
    I = cv2.filter2D(I, -1, D)
    plt.subplot(2,3,5)
    plt.imshow(I)
    plt.title("Gt, D")
    
    I = cv2.filter2D(impulse, -1, D.T)
    I = cv2.filter2D(I, -1, G)
    plt.subplot(2,3,6)
    plt.imshow(I)
    plt.title("Dt, G")

    plt.show()

##### 1d) #####
def naloga_1d():
    I = imread_gray("images/museum.jpg")
    Ix, Iy = first_derivative(I, 2)
    Ixx, Ixy, Iyx, Iyy = second_derivative(I, 2)
 
    plt.set_cmap("gray")
    plt.subplot(2, 4, 1)
    plt.imshow(I)
    plt.title("Original")
    
    plt.subplot(2, 4, 2)
    plt.imshow(Ix)
    plt.title("I_x")
    
    plt.subplot(2, 4, 3)
    plt.imshow(Iy)
    plt.title("I_y")
    
    plt.subplot(2, 4, 4)
    plt.imshow(Ixx)
    plt.title("I_mag")
    
    plt.subplot(2, 4, 5)
    plt.imshow(Iyx)
    plt.title("I_xx")
    
    plt.subplot(2, 4, 6)
    plt.imshow(Ixy)
    plt.title("I_xy")

    plt.subplot(2, 4, 7)
    plt.imshow(Iyy)
    plt.title("I_yy")

    plt.subplot(2, 4, 8)
    plt.imshow(Iyy)
    plt.title("I_dir")
    plt.show()

def first_derivative(I, sigma):
    gauss_kernel = gauss(sigma)
    gaussdx_kernel = gaussdx(sigma)

    Ix = cv2.filter2D(I.T, -1, gauss_kernel)
    Ix = Ix.T
    Ix = cv2.filter2D(Ix, -1, gaussdx_kernel)
    
    Iy = cv2.filter2D(I, -1, gauss_kernel)
    Iy = Iy.T
    Iy = cv2.filter2D(Iy, -1, gaussdx_kernel)
    return (Ix, Iy.T)

def second_derivative(I, sigma):
    Ix, Iy = first_derivative(I, sigma)
    Ixx, Ixy = first_derivative(Ix, sigma)
    Iyx, Iyy = first_derivative(Iy, sigma)
    return (Ixx, Ixy, Iyx, Iyy)

def gradient_magnitude(I, sigma):
    Ix, Iy = first_derivative(I, sigma)
    return (np.sqrt(Ix ** 2 + Iy ** 2), np.arctan2(Iy, Ix))

##### 1e) #####
def naloga_1e():
    I = imread("dataset/object_05_4.png")
    grad, mag = gradient_magnitude(I, 1)

def myhist(I, num_bins):
    H = np.zeros(num_bins)
    I = I.reshape(-1)
    # value_range = 1 / num_bins
    for i in I:
        ix = int(i * num_bins)
        # ix = int(i / value_range)    
        if ix < num_bins: H[ix] += 1
        else: H[num_bins - 1] += 1
        # print(ix, int)
    return H / len(I)

# retrieval_system
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
    ref_hist = myhist(ref_img, n_bins)
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
        plt.bar(range(n_bins * n_bins * n_bins), obj.hist, width = 5)
        plt.title(f"hell = {str(round(obj.hell, 2))}")
    plt.show()

def get_path_histograms(path, n_bins):
    hists = []
    for img_name in os.listdir(path): #[16:21]:
        name = f"dataset/{img_name}"
        I = imread(name)
        hist = myhist(I, n_bins)
        im = image_data(name, I, hist.flatten())
        hists.append(im)
    return hists

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


################################################
# Exercise 2: Edges in images

##### 2a) #####
def naloga_2a():
    I = imread_gray("images/museum.jpg")
    I = findedges(I, 1, 0.15)
    imshow(I)

def findedges(I, sigma, theta):
    mag, _ = gradient_magnitude(I, sigma)
    # print(mag)
    Ie = np.where(mag > theta, 1, 0)
    return Ie

##### 2b) #####
def naloga_2b():
    I = imread_gray("images/museum.jpg")
    imshow(suppression(I))

def suppression(I):
    magnitudes, angles = gradient_magnitude(I, 1)
    res = np.copy(magnitudes)
    for y in range(1, len(I) - 1):
        for x in range(1, len(I[1]) - 1):
            mag = magnitudes[y, x]
            a = (angles[y,x]) % np.pi
            # print(a)
            if(a >= 0 and a < np.pi / 4) and (mag < magnitudes[y, x - 1] or mag < magnitudes[y, x + 1]): res[y, x] = 0
            elif(a >= np.pi / 4 and a < np.pi / 2) and (mag < magnitudes[y + 1, x - 1] or mag < magnitudes[y - 1, x + 1]): res[y, x] = 0
            elif(a >= np.pi / 2 and a < 3 * np.pi / 4) and (mag < magnitudes[y - 1, x] or mag < magnitudes[y + 1, x]): res[y, x] = 0
            elif(a >= 3 * np.pi / 4 and a < np.pi) and (mag < magnitudes[y - 1, x - 1] or mag < magnitudes[y + 1, x + 1]): res[y, x] = 0
    return res

################################################
# Exercise 3: Detecting lines
##### 3a) #####
def naloga_3a():
    plt.subplot(2,2,1)
    plt.imshow(accumulator(300, 300, 10, 10))
    plt.title("x = 10, y = 10")

    plt.subplot(2,2,2)
    plt.imshow(accumulator(300, 300, 30, 60))
    plt.title("x = 30, y = 60")
    
    plt.subplot(2,2,3)
    plt.imshow(accumulator(300, 300, 50, 20))
    plt.title("x = 50, y = 20")
    
    plt.subplot(2,2,4)
    plt.imshow(accumulator(300, 300, 80, 90))
    plt.title("x = 80, y = 90")

    plt.show()

def accumulator(bins_theta, bins_rho, x, y):
    max_rho = 100
    val_theta = (np.linspace(-90, 90, bins_theta) / 180)  * np.pi
    rho_values = x * np.cos(val_theta) + y * np.sin(val_theta)
    rho = np.round(((rho_values + max_rho) / (2 * max_rho)) * (bins_rho - 1)).astype(int)

    accumulator = np.zeros((bins_rho, bins_theta))
    for i in range(bins_theta):
        if 0 <= rho[i] < bins_rho:
            accumulator[rho[i], i] += 1
    return accumulator

##### 3b) #####
def naloga_3b():
    I = np.zeros((100, 100))
    I[10, 10] = 1
    I[10, 20] = 1

    oneline = imread_gray("images/oneline.png")
    rectangle = imread_gray("images/rectangle.png")
    oneline = findedges(oneline, 1, 0.1)
    rectangle = findedges(rectangle, 1, 0.4)

    plt.set_cmap("viridis")
    plt.subplot(2, 3, 1)
    plt.imshow(hough_find_lines(I, 300, 300))#, 10))
    plt.title("Synthetic")

    plt.subplot(2, 3, 2)
    plt.imshow(hough_find_lines(oneline, 300, 300)) #, 10))
    plt.title("oneline.png")

    plt.subplot(2, 3, 3)
    plt.imshow(hough_find_lines(rectangle, 300, 300)) #, 10))
    plt.title("rectangle.png")

    plt.subplot(2, 3, 4)
    plt.imshow(I)
    plt.title("Synthetic")

    plt.subplot(2, 3, 5)
    plt.imshow(oneline)
    plt.title("oneline.png")

    plt.subplot(2, 3, 6)
    plt.imshow(rectangle)
    plt.title("rectangle.png")
    plt.show()

def hough_find_lines(I, bins_theta, bins_rho): #, threshold):
    A = np.zeros((bins_rho, bins_theta))
    D = int(np.sqrt(len(I) ** 2 + len(I[1]) ** 2))
    for y in range(len(I)):
        for x in range(len(I[1])):
            if I[y, x] > 0:
                A += accumulator(bins_theta, bins_rho, x, y)
    # print(A)
    return A

# ##### 3c) #####
def naloga_3c():
    I = np.zeros((100, 100))
    I[10, 10] = 1
    I[10, 20] = 1

    oneline = imread_gray("images/oneline.png")
    rectangle = imread_gray("images/rectangle.png")
    oneline = findedges(oneline, 1, 0.1)
    rectangle = findedges(rectangle, 1, 0.4)

    # plt.set_cmap("viridis")
    plt.subplot(2, 3, 1)
    plt.imshow(nonmaxima_suppression_box(hough_find_lines(I, 300, 300)))#, 10))
    plt.title("Synthetic")

    plt.subplot(2, 3, 2)
    plt.imshow(nonmaxima_suppression_box(hough_find_lines(oneline, 300, 300))) #, 10))
    plt.title("oneline.png")

    plt.subplot(2, 3, 3)
    plt.imshow(nonmaxima_suppression_box(hough_find_lines(rectangle, 300, 300))) #, 10))
    plt.title("rectangle.png")

    plt.subplot(2, 3, 4)
    plt.imshow(I)
    plt.title("Synthetic")

    plt.subplot(2, 3, 5)
    plt.imshow(oneline)
    plt.title("oneline.png")

    plt.subplot(2, 3, 6)
    plt.imshow(rectangle)
    plt.title("rectangle.png")
    plt.show()

def nonmaxima_suppression_box(I):
    J = np.copy(I)
    for i in range(1, len(I) - 1):
        for j in range(1, len(I[1]) - 1):
            box = J[i - 1:i + 1, j - 1:j + 1]  # 8-neighborhood
            # np.where(not J[i, j] >= box.all(), 0, I[i, j])
            if not J[i, j] >= box.all():
                J[i, j] = 0
    return J

# ##### 3d) #####
def naloga_3d():
    I = np.zeros((100, 100))
    I[10, 10] = 1
    I[10, 20] = 1
    oneline = imread_gray('images/oneline.png')
    rectangle = imread_gray('images/rectangle.png')
    
    oneline_edges = findedges(oneline, 1, 0.1)
    rectangle_edges = findedges(rectangle, 1, 0.4)

    I_lines = hough_find_lines(I, 100, 100)
    oneline_lines = hough_find_lines(oneline_edges, 450, 450)
    rectangle_lines = hough_find_lines(rectangle_edges, 450, 450)

    I_surpressed = nonmaxima_suppression_box(I_lines)
    oneline_surpressed = nonmaxima_suppression_box(oneline_lines)
    rectangle_surpressed = nonmaxima_suppression_box(rectangle_lines)

    threshold = 100
    r, t = np.where(I_surpressed > threshold)
    plt.subplot(1, 3, 1)
    plt.imshow(I)
    for rho, theta in zip(r, t):
        draw_line(rho, theta, len(I), len(I[1]))
    
    r, t = np.where(oneline_surpressed > threshold)
    plt.subplot(1, 3, 2)
    plt.imshow(oneline)
    for rho, theta in zip(r, t):
        draw_line(rho, theta, len(oneline), len(oneline[1]))
 
    r, t = np.where(rectangle_surpressed > threshold)
    plt.subplot(1, 3, 3)
    plt.imshow(rectangle)
    for rho, theta in zip(r, t):
        draw_line(rho, theta, len(rectangle), len(rectangle[1]))
    plt.show()
     
##### 3e) #####
def naloga_3e():
    bricks = imread("images/bricks.jpg")
    bricks_gray = cv2.cvtColor(cv2.imread("images/bricks.jpg"), cv2.COLOR_BGR2GRAY).astype('uint8')
    # bricks_gray = imread_gray("images/bricks.jpg")
        
    bricks_edges = cv2.Canny(bricks_gray, 400, 250, 50)
    bricks_lines = hough_find_lines(bricks_edges, 400, 400)
    bricks_supressed = nonmaxima_suppression_box(bricks_lines)
    # imshow(bricks_lines)
    # print(bricks_lines)
    n = 10
    bricks_sorted = np.argsort(bricks_supressed, axis = None)[-n:]    
    # print(bricks_sorted)
    plt.subplot(1, 2, 1)
    plt.imshow(bricks_supressed)
    for rho, theta in bricks_sorted:
        draw_line(rho, theta, len(bricks), len(bricks[1]))
    plt.show()


    pier = imread("images/pier.jpg")
    pier_gray = cv2.cvtColor(cv2.imread("images/pier.jpg"), cv2.COLOR_BGR2GRAY).astype('uint8')
    # pier_gray = imread_gray("images/pier.jpg")
    pier_edges = cv2.Canny(pier_gray, 400, 250, 50)
    pier_lines = hough_find_lines(pier_edges, 400, 400)
    pier_supressed = nonmaxima_suppression_box(pier_lines)
    n = 10
    pier_sorted = np.argsort(pier_supressed, axis = None)[-n:]    
    # print(pier_sorted)
    plt.subplot(1, 2, 2)
    plt.imshow(pier_supressed)
    for rho, theta in pier_sorted:
        draw_line(rho, theta, len(pier), len(pier[1]))
    plt.show()
    
##### 3f) #####
# def hough_find_lines(I, bins_theta, bins_rho):
def naloga_3f(I, bins_theta, bins_rho):
    A = np.zeros((bins_rho, bins_theta))
    D = int(np.sqrt(len(I) ** 2 + len(I[1]) ** 2))
    for y in range(len(I)):
        for x in range(len(I[1])):
            if I[y, x] > 0:
                A += accumulator(bins_theta, bins_rho, x, y)
    return A

##### 3g) #####
def naloga_3g(I, bins_theta, bins_rho):
    A = np.zeros((bins_rho, bins_theta))
    D = int(np.sqrt(len(I) ** 2 + len(I[1]) ** 2))
    for y in range(len(I)):
        for x in range(len(I[1])):
            if I[y, x] > 0:
                A += accumulator(bins_theta, bins_rho, x, y)
    return A


naloga_2a()