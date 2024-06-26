# COMMON ERRORS AND DEBUG IDEAS
# • check your x and y coordinates
# • check your data type: float, uint8
# • check your data range: [0,255], [0,1]
# • perform simple checks (synthetic data examples)

from a4_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import random


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
# Exercise 1: Feature points detectors

##### 1a) #####
def naloga_1a():
  I = imread_gray("data/graf/graf_a.jpg")
  thresh = 0.004
  sigma = 3

  det3 = hessian_points(I, 3)
  det6 = hessian_points(I, 6)
  det9 = hessian_points(I, 9)
  det3_suppressed = nonmaxima_suppression_box(det3, thresh)
  det6_suppressed = nonmaxima_suppression_box(det6, thresh)
  det9_suppressed = nonmaxima_suppression_box(det9, thresh)

  plt.subplot(2, 3, 1)
  plt.imshow(det3)
  plt.title("sigma = 3")

  plt.subplot(2, 3, 2)
  plt.imshow(det6)
  plt.title("sigma = 6")

  plt.subplot(2, 3, 3)
  plt.imshow(det9)
  plt.title("sigma = 9")

  plt.subplot(2, 3, 4)
  plt.imshow(I, cmap="gray")
  points = np.where(det3_suppressed == 1)
  plt.scatter(points[1], points[0], marker="x", s=1, c="r")
  
  plt.subplot(2, 3, 5)
  plt.imshow(I, cmap="gray")
  points = np.where(det6_suppressed == 1)
  plt.scatter(points[1], points[0], marker="x", s=1, c="r")
  
  plt.subplot(2, 3, 6)
  plt.imshow(I, cmap="gray")
  points = np.where(det9_suppressed == 1)
  plt.scatter(points[1], points[0], marker="x", s=1, c="r")

  plt.show()

def first_derivative(I, sigma):
  g = gauss(sigma)
  d = gaussdx(sigma)
  Ix = convolve(I, g.T, d)
  Iy = convolve(I, g, d.T)
  return (Ix, Iy)

def second_derivative(I, sigma):
  g = gauss(sigma)
  d = gaussdx(sigma)
  Ix = convolve(I, g.T, d)
  Iy = convolve(I, g, d.T)
  Ixx = convolve(Ix, g.T, d)
  Iyy = convolve(Iy, g, d.T)
  Ixy = convolve(Ix, g, d.T)
  return (Ixx, Ixy, Iyy)

def hessian_points(I, sigma):
  Ixx, Ixy, Iyy = second_derivative(I, sigma)
  return Ixx * Iyy - Ixy ** 2

def nonmaxima_suppression_box(det, threshold):
  res = np.zeros(det.shape)
  for y in range(1, len(det[0]) - 1):
    for x in range(1, len(det) - 1):
      box = det[x - 1 : x + 2, y - 1 : y + 2]
      if det[x, y] > threshold and det[x, y] == np.max(box): #and det[x, y] >= np.all(box):
        res[x, y] = 1
      # else:
      #   res[x, y] = 0
  return res

##### 1b) #####
def naloga_1b():
  I = imread_gray("data/graf/graf_a.jpg")
  thresh = 1e-6
  alpha = 0.06
  sigma_tilda = 1.6

  det3 = harris(I, 3, sigma_tilda, alpha)
  det6 = harris(I, 6, sigma_tilda, alpha)
  det9 = harris(I, 9, sigma_tilda, alpha)
  det3_suppressed = nonmaxima_suppression_box(det3, thresh)
  det6_suppressed = nonmaxima_suppression_box(det6, thresh)
  det9_suppressed = nonmaxima_suppression_box(det9, thresh)
  
  plt.subplot(2, 3, 1)
  plt.imshow(det3)
  plt.title("sigma = 3")

  plt.subplot(2, 3, 2)
  plt.imshow(det6)
  plt.title("sigma = 6")

  plt.subplot(2, 3, 3)
  plt.imshow(det9)
  plt.title("sigma = 9")

  plt.subplot(2, 3, 4)
  plt.imshow(I, cmap="gray")
  points = np.where(det3_suppressed == 1)
  plt.scatter(points[1], points[0], marker="x", s=1, c="r")
  
  plt.subplot(2, 3, 5)
  plt.imshow(I, cmap="gray")
  points = np.where(det6_suppressed == 1)
  plt.scatter(points[1], points[0], marker="x", s=1, c="r")
  
  plt.subplot(2, 3, 6)
  plt.imshow(I, cmap="gray")
  points = np.where(det9_suppressed == 1)
  plt.scatter(points[1], points[0], marker="x", s=1, c="r")

  plt.show()

def harris(I, sigma, sigma_tilda, alpha):
  Ix, Iy = first_derivative(I, sigma)
  g = gauss(sigma_tilda * sigma)
  C11 = convolve(Ix * Ix, g, g.T) 
  C12 = convolve(Ix * Iy, g, g.T)
  C22 = convolve(Iy * Iy, g, g.T)
  det = C11 * C22 - C12 * C12 # C21 == C12
  trace = C11 + C22
  return det - alpha * (trace ** 2)

################################################
# Exercise 2: Matching local regions

##### 2a) #####
def naloga_2a():
  image_a = imread_gray("data/graf/graf_a_small.jpg")
  image_b = imread_gray("data/graf/graf_b_small.jpg")
  thresh = 1e-6
  alpha = 0.06
  sigma_tilda = 1.6
  sigma = 6

  feature_points_a = nonmaxima_suppression_box(harris(image_a, sigma, sigma_tilda, alpha), thresh)
  feature_points_b = nonmaxima_suppression_box(harris(image_b, sigma, sigma_tilda, alpha), thresh)
  feature_positions_a = np.argwhere(feature_points_a) # returns positions with feature_points_a != 0 
  feature_positions_b = np.argwhere(feature_points_b)
  descriptors_a = simple_descriptors(image_a, feature_positions_a[:,0], feature_positions_a[:,1])
  descriptors_b = simple_descriptors(image_b, feature_positions_b[:,0], feature_positions_b[:,1])

  pairs = find_correspondences(descriptors_a, descriptors_b)  # indeksi
  a, b = [], []
  for p in pairs:
    a.append(feature_positions_a[p[0]])
    b.append(feature_positions_b[p[1]])
  a = np.flip(a, -1)  # flip ker display_matches predvideva y x
  b = np.flip(b, -1)
  # print(a, b)
  display_matches(image_a, a, image_b, b)

def find_correspondences(descriptors_a, descriptors_b):
  # hellinger = np.sqrt(np.sum((np.sqrt(h1) - np.sqrt(h2)) ** 2) / 2)
  hell = lambda x, y: np.sqrt(np.sum((np.sqrt(x) - np.sqrt(y)) ** 2, -1) / 2)
  pairs = []
  for ix_a, descriptor_a in enumerate(descriptors_a):
    pairs.append( (ix_a, np.argmin(hell(descriptor_a, descriptors_b))) ) # finds the index of closest descriptor
  return pairs

##### 2b) #####
def naloga_2b():
  image_a = imread_gray("data/graf/graf_a_small.jpg")
  image_b = imread_gray("data/graf/graf_b_small.jpg")
  a, b = find_matches(image_a, image_b)
  display_matches(image_a, a, image_b, b)

def find_matches(image_a, image_b):
  thresh = 1e-6
  alpha = 0.06
  sigma_tilda = 1.6
  sigma = 6

  feature_points_a = nonmaxima_suppression_box(harris(image_a, sigma, sigma_tilda, alpha), thresh)
  feature_points_b = nonmaxima_suppression_box(harris(image_b, sigma, sigma_tilda, alpha), thresh)
  feature_positions_a = np.argwhere(feature_points_a) # returns positions with feature_points_a != 0 
  feature_positions_b = np.argwhere(feature_points_b)
  
  descriptors_a = simple_descriptors(image_a, feature_positions_a[:,0], feature_positions_a[:,1])
  descriptors_b = simple_descriptors(image_b, feature_positions_b[:,0], feature_positions_b[:,1])

  pairs1 = find_correspondences(descriptors_a, descriptors_b)  # indeksi
  pairs2 = find_correspondences(descriptors_b, descriptors_a)
  matches = []
  a, b = [], []
  for p in pairs1:
    pp = (p[1], p[0]) # preveri zamenjane vrednosti v drugih parih
    if pp in pairs2:
      matches.append(p)
      a.append(feature_positions_a[p[0]])
      b.append(feature_positions_b[p[1]])
  a = np.flip(a, -1)
  b = np.flip(b, -1)
  # print(a, b)
  return a, b

##### 2c) #####
# Correspondences using keypoints
#  Strategy 3: Calculate the distance between A and
#  - the second-most similar keypoint in the right image
#  - the most similar keypoint in the right image.
#  Ratio of these two distances (first/second) will be low for distinctive key-points and high for non-distinctive ones.
#  Threshold ~0.8 gives good results with SIFT.
def naloga_2c():
    image_a = imread_gray("data/graf/graf_a_small.jpg")
    image_b = imread_gray("data/graf/graf_b_small.jpg")
    a, b = find_matches(image_a, image_b)
    thresh = 1e-6
    alpha = 0.06
    sigma_tilda = 1.6
    sigma = 6

    feature_points_a = nonmaxima_suppression_box(harris(image_a, sigma, sigma_tilda, alpha), thresh)
    feature_points_b = nonmaxima_suppression_box(harris(image_b, sigma, sigma_tilda, alpha), thresh)
    feature_positions_a = np.argwhere(feature_points_a) # returns positions with feature_points_a != 0 
    feature_positions_b = np.argwhere(feature_points_b)
    descriptors_a = simple_descriptors(image_a, feature_positions_a[:,0], feature_positions_a[:,1])
    descriptors_b = simple_descriptors(image_b, feature_positions_b[:,0], feature_positions_b[:,1])

    hell = lambda x, y: np.sqrt(np.sum((np.sqrt(x) - np.sqrt(y)) ** 2, -1) / 2)
    ratio = 0.8
    matches = []
    for ix_a, descriptor_a in enumerate(descriptors_a):
      distances = hell(descriptor_a, descriptors_b)
      sorted_indices = np.argsort(distances)
      if distances[sorted_indices[0]] / distances[sorted_indices[1]] < ratio:
        matches.append( (feature_positions_a[ix_a], feature_positions_b[sorted_indices[0]]) )
    a, b = zip(*matches)
    a = np.flip(np.array(a), -1)
    b = np.flip(np.array(b), -1)
    display_matches(image_a, a, image_b, b)

##### 2e) #####
def naloga_2e(): # https://stackabuse.com/real-time-pose-estimation-from-video-in-python-with-yolov7/
  video = cv2.VideoCapture("data/video.mp4")
  sift = cv2.SIFT_create()
  while video.isOpened():
    ret, frame = video.read()
    if not ret: break
    keypoints = sift.detect(frame, None)
    keypoints_frame = cv2.drawKeypoints(frame, keypoints, frame)  # drawKeypoints(input_image, key_points, output_image, colour, flag)
    cv2.imshow("video", keypoints_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break
  video.release()
  cv2.destroyAllWindows()

################################################
# Exercise 3: Homography estimation

##### 3a) #####
def naloga_3a():
  newyork_a = imread_gray("data/newyork/newyork_a.jpg")
  newyork_b = imread_gray("data/newyork/newyork_b.jpg")
  feature_points = np.loadtxt("data/newyork/newyork.txt")
  # print(feature_points)
  # display_matches(newyork_a, feature_points[:, (0, 1)], newyork_b, feature_points[:, (2, 3)])
  H = estimate_homography(feature_points)
  newyork_a_warped = cv2.warpPerspective(newyork_a, H, newyork_a.shape) # cv2.warpPerspective(src, dst, dsize)

  plt.set_cmap("gray")
  plt.subplot(1, 3, 1)
  plt.imshow(newyork_a)
  plt.title("newyork_a")
  plt.subplot(1, 3, 2)
  plt.imshow(newyork_b)
  plt.title("newyork_b")
  plt.subplot(1, 3, 3)
  plt.imshow(newyork_a_warped)
  plt.title("transformed newyork_a")
  plt.show()

def estimate_homography(feature_points):
    A = np.zeros((2 * len(feature_points), 9))
    for i, p in enumerate(feature_points):
      xr, yr, xt, yt = p
      A[2 * i] = [xr, yr, 1, 0, 0, 0, -(xt * xr), -(xt * yr), -xt]
      A[2 * i + 1] = [0, 0, 0, xr, yr, 1, -(yt * xr), -(yt * yr), -yt]

    U, S, VT = np.linalg.svd(A)
    h = VT[-1:] # vector h is obtained from the last column of matrix VT
    H = h.reshape((3, 3))
    return H

##### 3b) #####
def naloga_3b():
  newyork_a = imread_gray("data/newyork/newyork_a.jpg")
  newyork_b = imread_gray("data/newyork/newyork_b.jpg")

  matching_points_a, matching_points_b = find_matches(newyork_a, newyork_b)
  # display_matches(newyork_a, matching_points_a, newyork_b, matching_points_b)
  H = ransac(matching_points_a, matching_points_b)
  newyork_a_warped = cv2.warpPerspective(newyork_a, H, newyork_a.shape) # cv2.warpPerspective(src, dst, dsize)

  plt.set_cmap("gray")
  plt.subplot(1, 2, 1)
  plt.imshow(newyork_b)
  plt.title("newyork_b")
  plt.subplot(1, 2, 2)
  plt.imshow(newyork_a_warped)
  plt.title("transformed newyork_a")
  plt.show()

def ransac(matching_points_a, matching_points_b):
  k = 1000 # num of iterations
  num_matches = 10
  reprojection_error = 100

  matching_points = np.hstack((matching_points_a, matching_points_b))
  best_H = []
  for iteration in range(k):
    random_points = random.sample(range(len(matching_points_a)), num_matches)
    random_sample = matching_points[random_points]
    H = estimate_homography(random_sample)

    inliers = []
    for match in matching_points:
      x1, y1, x2, y2 = match
      point_a = np.array([[x1], [y1], [1]])
      point_b = np.array([[x2], [y2], [1]])
      error = np.linalg.norm(np.dot(H, point_a) - point_b)
      if error < reprojection_error:
        inliers.append(match)

    if len(inliers) > len(best_H):
      best_H = H
      # print(best_H)

  return best_H


naloga_2e()