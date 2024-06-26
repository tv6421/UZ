from a5_utils import *
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
# Exercise 1: Disparity

##### 1a) #####
def naloga_1a():
  pass
##### 1b) #####
def naloga_1b():
  f = 2.5 / 1000  # 2.5 mm
  T = 12 / 100    # 12 cm
  p_z = np.arange(0, 1, 0.05)
  d = (f * T) / p_z
  plt.plot(p_z, d)
  plt.show()

##### 1c) #####
def naloga_1c():
  pass

##### 1d) #####
def naloga_1d():
  print(ncc(2, 5))

def ncc(X, Y):
  res = np.sum((X - np.mean(X)) * (Y - np.mean(Y))) \
        / np.sqrt(np.sum((X - np.mean(X)) ** 2) * np.sum((Y - np.mean(Y)) ** 2))
  return res

##### 1e) #####
def naloga_1e():
  pass

################################################
# Exercise 2: Fundamental matrix, epipoles, epipolar lines
  
##### 2a) #####
def naloga_2a():
  pass

##### 2b) #####
def naloga_2b():
  house1 = imread_gray("data/epipolar/house1.jpg")
  house2 = imread_gray("data/epipolar/house2.jpg")
  house_points = np.loadtxt("data/epipolar/house_points.txt")
  F = fundamental_matrix(house_points)

  # Once we have the fundamental matrix, we can take any point x in the first image
  # plane and determine an epipolar line for that point in the second image plane l' = Fx.
  plt.subplot(1, 2, 1)
  plt.imshow(house1, cmap="gray")
  for point in house_points:
    x1, y1, x2, y2 = point
    x_ = [x2, y2, 1]
    l = np.dot(F.T, x_)
    plt.plot(x1, y1, "o", color="red", markersize=3)
    draw_epiline(l, house1.shape[0], house1.shape[1])

  plt.subplot(1, 2, 2)
  plt.imshow(house2, cmap="gray")
  for point in house_points:
    x1, y1, x2, y2 = point
    x = [x1, y1, 1]
    l_ = np.dot(F, x)
    plt.plot(x2, y2, "o", color="red", markersize=3)
    draw_epiline(l_, house2.shape[0], house2.shape[1])
  plt.show()

def fundamental_matrix(house_points):    
  transformed_points1, T1 = normalize_points(house_points[:, 0:2])
  transformed_points2, T2 = normalize_points(house_points[:, 2:4])
  transformed_points = np.hstack((transformed_points1, transformed_points2))
  # Construct the matrix A as in equation  
  A = np.zeros((len(house_points), 9))
  for i, point in enumerate(transformed_points):
    u, v, x, u_, v_, x = point  # x = 1
    A[i] = [u * u_, u_ * v, u_, u * v_, v * v_, v_, u, v, 1]
  # Decompose the matrix using SVD ... and transform the last eigenvector ...
  U, D, V_T = np.linalg.svd(A)
  last_eigenvector = V_T.T[:, 8] # print(last_eigenvector)
  F_hat = last_eigenvector.reshape((3,3))
  # Decompose F = UDVT and set the lowest eigenvalue to 0, reconstruct F = UDVT
  U, D, V_T = np.linalg.svd(F_hat)
  D[-1] = 0
  F_hat = np.dot(np.dot(U, np.diag(D)), V_T)
  F = np.dot(np.dot(T2.T, F_hat), T1)
  return F

##### 2c) #####
def naloga_2c():
  house_points = np.loadtxt("data/epipolar/house_points.txt")
  F = fundamental_matrix(house_points)

  point_error = reprojection_error(F, [85, 233, 67, 219])
  print("REPROJECTION ERROR for p1 = [85, 233] and p2 = [67, 219]:", point_error)
  
  points_error = 0
  for point in house_points: 
    points_error += reprojection_error(F, point)
  points_error /= len(house_points)
  print("REPROJECTION ERROR for house_points:", points_error)

def reprojection_error(F, point):
  x1, y1, x2, y2 = point
  x = [x1, y1, 1]
  x_ = [x2, y2, 1]
  l = np.dot(F.T, x_)
  l_ = np.dot(F, x)
  
  # distance for epipolar line l
  a, b, c = l[0], l[1], l[2]
  distance_l = np.abs(a * x1 + b * y1 + c) / (np.sqrt(a ** 2 + b ** 2))
  # distance for epipolar line l'
  a, b, c = l_[0], l_[1], l_[2]
  distance_l_ = np.abs(a * x2 + b * y2 + c) / (np.sqrt(a ** 2 + b ** 2))
  return (distance_l + distance_l_) / 2

##### 2d) #####
def naloga_2d():
  def ransac(matching_points_a, matching_points_b):
    k = 100  #naloga_3c() # num of iterations
    num_matches = 4
    reprojection_error = 150

    matching_points = np.hstack((matching_points_a, matching_points_b))
    best_H = []
    best_error = 100000
    for iteration in range(k):
      random_points = random.sample(range(len(matching_points_a)), num_matches)
      random_sample = matching_points[random_points]
      H = estimate_homography(random_sample)

      inliers = []
      sum_error = 0
      for match in matching_points:
        x1, y1, x2, y2 = match
        point_a = np.array([[x1], [y1], [1]])
        point_b = np.array([[x2], [y2], [1]])
        a = np.dot(H, point_a)
        a = a / a[-1]
        error = np.linalg.norm(a - point_b)
        if error < reprojection_error:
          inliers.append(match)
          sum_error += error

      if len(inliers) / len(matching_points) > 0.2: # inlier percentage
        H = estimate_homography(inliers)
        if sum_error < best_error:
          best_H = H
          best_error = sum_error

      print(best_error)
    return best_H
  
  def estimate_homography(feature_points):
    # H * hr = ht, preoblikujes v sistem enacb z 8 parametri
    A = np.zeros((2 * len(feature_points), 9))
    for i, p in enumerate(feature_points):
      # so to enacbo dobis A matriko
      xr, yr, xt, yt = p
      A[2 * i] = [xr, yr, 1, 0, 0, 0, -(xt * xr), -(xt * yr), -xt]
      A[2 * i + 1] = [0, 0, 0, xr, yr, 1, -(yt * xr), -(yt * yr), -yt]

    # iz A matrike z SVD razcepom dobis V
    U, S, V = np.linalg.svd(A)
    h = V[-1:] # vector h is obtained from the last column of matrix V_T
    H = h.reshape((3, 3))
    return H


################################################
# Exercise 3: Triangulation
  
##### 3a) #####
def naloga_3a():
  house1 = imread_gray("data/epipolar/house1.jpg")
  house2 = imread_gray("data/epipolar/house2.jpg")

  house_points = np.loadtxt("data/epipolar/house_points.txt")
  P1 = np.loadtxt("data/epipolar/house1_camera.txt")
  P2 = np.loadtxt("data/epipolar/house2_camera.txt")
  triangulated_points = triangulate(house_points, P1, P2)

  plt.figure(figsize = (10, 5))
  plt.subplot(1, 3, 1)
  plt.imshow(house1, cmap="gray")
  for point_ix, point in enumerate(house_points):
    x1, y1, x2, y2 = point
    plt.plot(x1, y1, "o", color="red", markersize=3)
    plt.text(x1, y1, point_ix)

  plt.subplot(1, 3, 2)
  plt.imshow(house2, cmap="gray")
  for point_ix, point in enumerate(house_points):
    x1, y1, x2, y2 = point
    plt.plot(x2, y2, "o", color="red", markersize=3)
    plt.text(x2, y2, point_ix)

  T = np.array([[-1, 0,  0],
                [ 0, 0, -1],
                [ 0, 1,  0]])
  ax = plt.subplot(1, 3, 3, projection="3d")
  for point_ix, point in enumerate(triangulated_points):
    x, y, z = T.dot(point[0:3]) # point[3] == 1
    ax.plot(x, y, z, "o", color="red", markersize=3)
    ax.text(x, y, z, s=point_ix) # s je za text
  plt.show()

def triangulate(house_points, P1, P2):
  triangulated_points = np.zeros((len(house_points), 4))
  for point_ix, point in enumerate(house_points):
    x1, y1, x2, y2 = point
    x1x = np.array([[0, -1, y1], # a_z = 1
                     [1, 0, -x1],
                     [-y1, x1, 0]])
    x2x = np.array([[0, -1, y2], # a_z = 1
                     [1, 0, -x2],
                     [-y2, x2, 0]])
    # combine only the first two lines (2x3) and stack them (4x3)
    x1x_P1 = np.dot(x1x, P1)
    x2x_P2 = np.dot(x2x, P2)
    A = np.vstack((x1x_P1[0:2], x2x_P2[0:2]))
    # compute estimate of X
    U, D, V_T = np.linalg.svd(A)
    V_T = V_T.T
    X = V_T[:,-1] / V_T[-1, -1]
    triangulated_points[point_ix] = X
  return triangulated_points


naloga_2b()