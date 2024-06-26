from a6_utils import *
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
# Exercise 1: Direct PCA method
    
##### 1a) #####
def naloga_1a():
  X = np.array([(3, 4), (3, 6), (7, 6), (6, 4)])
  # 2: Calculate the mean
  mi = np.mean(X, axis = 0)
  # 3: Center the data
  X_d = X - mi
  # 4: Compute the covariance matrix
  C = (X_d.T @ X_d) / (len(X) - 1)
  # 5: Compute SVD of the covariance matrix
  U, S, V_T = np.linalg.svd(C)
  print("C: ", C, "U: ", U, "S: ", S, "VT: ", V_T)
  print("eigenvalues: ", S, "eigenvectors: \n", V_T) 

  ax = plt.subplot()
  for point_ix, point in enumerate(X):
    x, y = point
    ax.plot(x, y, "o", color="red", markersize=3)
    ax.text(x, y, s = point_ix + 1) # s je za text
  
  eigenvector1 = V_T.T[:,0] * np.sqrt(S[0]) + mi
  eigenvector2 = V_T.T[:,1] * np.sqrt(S[1]) + mi
  plt.plot((0, 0), (V_T[0][0], V_T[0][1]), 'r')
  plt.plot((mi[0], eigenvector2[0]), (mi[1], eigenvector2[1]), 'g')
  plt.axis("equal")
  plt.show()

##### 1b) + 1c) #####
def naloga_1bc():
  X = np.loadtxt("data/points.txt")
  # 2: Calculate the mean
  mi = np.mean(X, axis = 0)
  # 3: Center the data
  X_d = X - mi
  # 4: Compute the covariance matrix
  C = (X_d.T @ X_d) / (len(X) - 1)
  # 5: Compute SVD of the covariance matrix
  U, S, V_T = np.linalg.svd(C)

  plt.figure(figsize = (15, 7))
  plt.subplot(1, 2, 1)
  
  ax = plt.subplot()
  for point_ix, point in enumerate(X):
    x, y = point
    ax.plot(x, y, "o", color="red", markersize = 3)
    ax.text(x, y, s = point_ix + 1) # s je za text
  
  eigenvector1 = V_T.T[:,0] * np.sqrt(S[0]) + mi
  eigenvector2 = V_T.T[:,1] * np.sqrt(S[1]) + mi
  plt.plot((mi[0], eigenvector1[0]), (mi[1], eigenvector1[1]), 'r')
  plt.plot((mi[0], eigenvector2[0]), (mi[1], eigenvector2[1]), 'g')
  drawEllipse(mi, C, 1)
  
  # plt.axis("equal")
  plt.show()

##### 1d) #####
def naloga_1d():
  X = np.loadtxt("data/points.txt")
  # 2: Calculate the mean
  mi = np.mean(X, axis = 0)
  # 3: Center the data
  X_d = X - mi
  # 4: Compute the covariance matrix
  C = (X_d.T @ X_d) / (len(X) - 1)
  # 5: Compute SVD of the covariance matrix
  U, S, V_T = np.linalg.svd(C)

  cumulative = np.cumsum(S)
  cumulative /= cumulative[1] # normalize
  print(f"variance %: {cumulative[0]}")
  plt.bar(range(0, 2), cumulative)
  plt.show()

##### 1e) + 1f) #####
def naloga_1ef():
  X = np.loadtxt("data/points.txt")
  # 2: Calculate the mean
  mi = np.mean(X, axis = 0)
  # 3: Center the data
  X_d = X - mi
  # 4: Compute the covariance matrix
  C = (X_d.T @ X_d) / (len(X) - 1)
  # 5: Compute SVD of the covariance matrix
  U, S, V_T = np.linalg.svd(C)

  U[1:] = 0  # remove all vectors except the first one
  U = np.array(U)
  print("U: ", U)
  projected_points = (U.T @ (U @ X_d.T)).T + mi
  point = np.array([6, 6])
  y = U @ (point - mi)
  result = (U.T @ y) + mi

  # find the closest points using argmin
  closest_point = X[np.argmin(np.linalg.norm(X - point, axis=1))]  # evklidska razdalja
  print("closest:", closest_point)
  closest_point = projected_points[np.argmin(np.linalg.norm(projected_points - result, axis=1))]
  print("closest projected:", closest_point)

  # zacetne tocke
  plt.figure(figsize = (10, 5))
  ax = plt.subplot()
  for point_ix, point1 in enumerate(X):
    x, y = point1
    ax.plot(x, y, "o", color="blue", markersize = 3)
    ax.text(x, y, s = point_ix + 1) # s je za text

  # projectane tocke
  for point_ix, point2 in enumerate(projected_points):
    x, y = point2
    ax.plot(x, y, "o", color="black", markersize = 3)
    ax.text(x, y, s = f"{point_ix + 1}'") # s je za text

  x, y = point
  plt.plot(x, y, "o", color="orange", markersize = 3)
  ax.text(x, y, s = "q_point")
  x, y = result
  plt.plot(x, y, "o", color="orange", markersize = 3)
  ax.text(x, y, s = "q_point'")
 
  eigenvector1 = V_T.T[:,0] * np.sqrt(S[0]) + mi
  eigenvector2 = V_T.T[:,1] * np.sqrt(S[1]) + mi
  plt.plot((mi[0], eigenvector1[0]), (mi[1], eigenvector1[1]), 'r')
  plt.plot((mi[0], eigenvector2[0]), (mi[1], eigenvector2[1]), 'g')
  drawEllipse(mi, C, 1)

  plt.axis("equal")
  plt.show()

################################################
# Exercise 2: The dual PCA method
  
##### 2a) + 2b) #####
def naloga_2ab():
  X = np.loadtxt("data/points.txt")
  # 2: Calculate the mean
  mi = np.mean(X, axis = 0)
  # 3: Center the data
  X_d = X - mi
  # 4: Compute the DUAL covariance matrix
  C = (X_d @ X_d.T) / (len(X) - 1)

  # 5: Compute SVD of the DUAL covariance matrix
  U, S, V_T = np.linalg.svd(C)
  U = (X_d.T @ U) * np.sqrt(1 / (S * (len(X) - 1)))
  Y = X_d @ U
  x_q = (U @ Y.T).T + mi

  print("X - x_q:", X - x_q)  
  plt.plot(x_q[:, 0], x_q[:, 1], "o", color="blue", markersize = 5)
  plt.plot(X[:,0], X[:,1], "x", color="red", markersize = 5)
  plt.show()

################################################
# Exercise 3: Image decomposition examples

##### 3a) #####
def naloga_3a():
  image_vectors = []
  for p in os.listdir(f"data/faces/1"):
    I = imread_gray(f"data/faces/1/{p}")
    image_vectors.append(I.reshape(-1))
  return image_vectors

##### 3b) #####
def naloga_3b():
  image_vectors = naloga_3a()
  shape = (96, 84)
  U, mi = dual_pca(image_vectors)
  for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(np.reshape(U[:, i], shape), cmap="gray")
  plt.show()
  
  # Project the first image from the series to the PCA space and then back again
  I1_original = np.copy(image_vectors[0])
  y = U.T @ (I1_original - mi)
  I1_pca = (U @ y) + mi
  print("pca: ", np.linalg.norm(I1_original - I1_pca))
  I1_pca = I1_pca.reshape(shape)

  plt.figure(figsize = (12, 5))
  plt.subplot(1, 4, 1)
  plt.imshow(I1_original.reshape(shape), cmap="gray")
  plt.title("original")

  plt.subplot(1, 4, 2)
  plt.imshow(I1_pca, cmap="gray")
  plt.title("pca")
  
  # component with index 4074 to 0
  I1_original = np.copy(image_vectors[0])
  I1_original[4074] = 0
  y = U.T @ (I1_original - mi)
  I1_pca = (U @ y) + mi
  print("index 4070 = 0: ", np.linalg.norm(I1_original - I1_pca))
  I1_pca = I1_pca.reshape(shape)

  plt.subplot(1, 4, 3)
  plt.imshow(I1_pca, cmap="gray")
  plt.title("index 4070 = 0")

  # second vector = 0
  I1_original = np.copy(image_vectors[0])
  y = U.T @ (I1_original - mi)
  y[1] = 0 # najvecja razlika pri 1
  I1_pca = (U @ y) + mi
  print("second vector = 0: ", np.linalg.norm(I1_original - I1_pca))
  I1_pca = I1_pca.reshape(shape)

  plt.subplot(1, 4, 4)
  plt.imshow(I1_pca, cmap="gray")
  plt.title("second vector = 0")
  plt.show()

def dual_pca(X):
  # 2: Calculate the mean
  mi = np.mean(X, axis = 0)
  # 3: Center the data
  X_d = X - mi
  # 4: Compute the DUAL covariance matrix
  C = (X_d @ X_d.T) / (len(X) - 1)

  # 5: Compute SVD of the DUAL covariance matrix
  U, S, V_T = np.linalg.svd(C)
  U = (X_d.T @ U) * np.sqrt(1 / ((S + 10e-15) * (len(X) - 1)))
  return U, mi


##### 3c) #####
def naloga_3c():
  image_vectors = naloga_3a()
  shape = (96, 84)
  I1_original = np.copy(image_vectors[0])
  U, mi = dual_pca(image_vectors)
  y = U.T @ (I1_original - mi)

  num_of_eigenvectors = (32, 16, 8, 4, 2, 1)
  plt.figure(figsize = (15, 5))
  print(y)
  for i, num_components in enumerate(num_of_eigenvectors):
    y[num_components:] = 0
    I1_pca = (U @ y) + mi
    I1_pca = I1_pca.reshape(shape)
    plt.subplot(1, 6, i + 1)
    plt.imshow(I1_pca, cmap="gray")
    plt.title(f"{num_components}")    
  plt.show()


naloga_1a()
naloga_1bc()
naloga_1d()
naloga_1ef()
# naloga_2ab()
# naloga_3a()
# naloga_3b()
# naloga_3c()