---
title: "K-means Clustering and Principal Component Analysis"
date: 2018-02-21
tags: ["Python", "machine learning", "matplotlib"]
draft: false
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\[','\]']],
    processEscapes: true,
    processEnvironments: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    TeX: { equationNumbers: { autoNumber: "AMS" },
         extensions: ["AMSmath.js", "AMSsymbols.js"] }
  }
});
</script>

<script type="text/x-mathjax-config">
  MathJax.Hub.Queue(function() {
    // Fix <code> tags after MathJax finishes running. This is a
    // hack to overcome a shortcoming of Markdown. Discussion at
    // https://github.com/mojombo/jekyll/issues/199
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>

This post covers the first part of the seventh exercise from Andrew Ng's Machine Learning Course on Coursera.

***

# Introduction

The K-means clustering algorithm will be implemented and applied to compress an image. In a second step, principal component analysis will be used to find a low-dimensional representation of face images.

# K-means Clustering

K-means algorithm will be used for image compression. First, K-means algorithm will be applied in an example 2D dataset to help gain an intuition of how the algorithm works. After that, the K-means algorithm will be used for image compression by reducing the number of colours that occur in an image to only those that are most common in that image.

## Implementing K-means

The K-means algorithm is a method to automatically cluster similar data examples together. Concretely, a given training set `$\left\{x^{(1)},\dots,x^{(m)}\right\} \left(\text{ where } x^{(i)} \in \mathbb{R}^n \right)$` will be grouped into a few cohesive "clusters". The intuition behind K-means is an iterative procedure that starts by guessing the initial centroids, and then refines this guess by repeatedly assigning examples to their closest centroids and then recomputing the centroids based on the assignments.

The inner-loop of the algorithm repeatedly carries out two steps:

1. Assigning each training example `$x^{(i)}$` to its closest centroid, and
2. Recomputing the mean of each centroid using the points assigned to it.

The K-means algorithm will always converge to some final set of means for the centroids. Note that the converged solution may not always be ideal and depends on the initial setting of the centroids. Therefore, in practice the K-means algorithm is usually run a few times with different random initializations. One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (**distortion**).

### Finding Closest Centroids

In the "cluster assignment" phase of the K-means algorithm, the algorithm assigns every training example `$x^{(i)}$` to its closest centroid, given the current positions of centroids. Specifically, for every example `$i$` it is set

`$$c^{(i)} := j \text{ that minimizes } \lVert x^{(i)}-\mu_j \rVert^2,$$`

where `$c^{(i)}$` is the index of the centroid that is closest to `$x^{(i)}$`, and `$\mu_j$` is the position (value) of the j'th centroid. Note that `$c^{(i)}$` corresponds to `$idx[i]$` in the code in `findClosestCentroids`. This function takes the data matrix `$X$` and the locations of all centroids inside `centroids` and should output a one-dimensional array `$idx$` that holds the index (a value in `$\left\{1,...,K\right\}$`, where `$K$` is total number of centroids) of the closest centroid to every training example. This can be implemented by using a loop over every training example and every centroid.


```python
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math

# Load dataset.
data = loadmat('ex7data2.mat')
X = data["X"]

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Create a function to find the closest centroids.
def findClosestCentroids(X, centroids):
    """
    Returns the closest centroids in idx for a dataset X
    where each row is a single example. idx = m x 1 vector
    of centroid assignments (i.e. each entry in range [1..K])
    Args:
        X        : array(# training examples, 2)
        centroids: array(K, 2)
    Returns:
        idx      : array(# training examples, 1)
    """
    # Set K size.
    K = centroids.shape[0]

    # Initialise idx.
    idx = np.zeros((X.shape[0], 1), dtype=np.int8)

    # Iterate over every example, find its closest centroid, and store
    # the index inside idx at the appropriate location. Concretely,
    # idx[i] should contain the index of the centroid closest to
    # example i. Hence, it should be a value in the range 1..K.
    
#     # Iterate over training examples.
#     for i in range(X.shape[0]):
#         # Set norm distance to infinity.
#         min_dst = math.inf
#         # Iterate over centroids.
#         for k in range(K):
#             # Compute the norm distance.
#             dst = np.linalg.norm(X[i,:] - centroids[k,:], axis=0)
#             if dst < min_dst:
#                 min_dst = dst
#                 idx[i] = k
    
    # Alternative partial vectorized solution.
    # Iterate over training examples.
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        # argmin returns the indices of the minimum values along an axis,
        # replacing the need for a for-loop and if statement.
        min_dst = np.argmin(distances)
        idx[i] = min_dst
    
    return idx


# Find the closest centroids for the examples.
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(idx[:3])
print('\n(The closest centroids should be 0, 2, 1 respectively)')
```

    Closest centroids for the first 3 examples: 
    
    [[0]
     [2]
     [1]]
    
    (The closest centroids should be 0, 2, 1 respectively)


### Computing Centroid Means

Given assignments of every point to a centroid, the second phase of the algorithm recomputes, for each centroid, the mean of the points that were assigned to it. Specifically, for every centroid k it is set

`$$ \mu_k := \frac{1}{\left|C_k\right|} \sum_{i \in C_k} x^{(i)} $$`

where `$C_k$` is the set of examples that are assigned to centroid `$k$`. Concretely, if two examples say `$x^{(3)}$` and `$x^{(5)}$` are assigned to centroid `$k = 2$`, then it should be updatet `$\mu_2 = \frac{1}{2} \left(x^{(3)} + x^{(5)}\right)$`.

The code in `computeCentroids` implements this function using a loop over the centroids. The code may run faster if it uses a vectorized implementation instead of a loop over the examples.


```python
# Create a function to compute the new centroids.
def computeCentroids(X, idx, K):
    """
    Returns the new centroids by computing the means
    of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point,
    a vector idx of centroid assignments (i.e. each entry
    in range [1..K]) for each example, and K, the number of
    centroids. A matrix centroids is returned, where each row
    of centroids is the mean of the data points assigned to it.
    Args:
        X        : array(# training examples, 2)
        idx      : array(# training examples, 1)
        K        : int, # of centroids
    Returns:
        centroids: array(# of centroids, 2)
    """
    # Create useful variables
    m, n = X.shape
    
    # Initialize centroids matrix.
    centroids = np.zeros((K, n))
    # Iterate over every centroid and compute mean of all points that
    # belong to it. Concretely, the row vector centroids[k,:] should
    # contain the mean of the data points assigned to centroid k.
    
#     # Iterate over centroids.
#     for k in range(K):
#         # Iterate over training examples.
#         for i in range(m):
#             if idx[i] == k:
#                 points = X[i]
#                 centroids[k] = np.mean(points, axis=0)
    
    # Alternative partial vectorized solution.
    for k in range(K):
        centroids[k, :] = np.mean(X[idx.ravel() == k, :], axis=0)
    
    return centroids

# Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(centroids)
print('\nThe centroids should be:\n')
print('[ 2.42830111  3.15792418 ]')
print('[ 5.81350331  2.63365645 ]')
print('[ 7.11938687  3.6166844 ]')
```

    Centroids computed after initial finding of closest centroids: 
    
    [[ 2.42830111  3.15792418]
     [ 5.81350331  2.63365645]
     [ 7.11938687  3.6166844 ]]
    
    The centroids should be:
    
    [ 2.42830111  3.15792418 ]
    [ 5.81350331  2.63365645 ]
    [ 7.11938687  3.6166844 ]


## K-means on Example Dataset

After implementing the two functions (`findClosestCentroids` and `computeCentroids`), the next step is to run the K-means algorithm on a toy 2D dataset to see how K-means works. The functions are called from inside the `runKmeans` script. Notice that the code calls the two functions in a loop.

A visualization of the progress of the algorithm at each iteration is shown in the next figure.


```python
# Create a function to plot the data points.
def plotDataPoints(X, idx, K):
    """
    Plots data points in X, coloring them so that those 
    with the same index assignments in idx have the same color
    Args:
        X  : array(# training examples, 2)
        idx: array(# training examples, 1)
        K  : int, # of centroids
    """
    # Create a colors list.
    colors = [plt.cm.tab20(float(i) / 10) for i in idx]

    # Plot the data.
    plt.scatter(X[:,0], X[:,1], c=colors, alpha=0.5, s=2)

# Create a function to display the progress of K-Means.
def plotProgresskMeans(X, centroids, previous, idx, K, i):
    """
    Plots the data points with colors assigned to each centroid.
    With the previous centroids, it also plots a line between the
    previous locations and current locations of the centroids.
    Args:
        X        : array(# training examples, 2)
        centroids: array(# of centroids, 2)
        previous : array(# of centroids, 2)
        idx      : array(# training examples, 1)
        K        : int, # of centroids
        i        : # of iterations
    """
    # Plot the examples.
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's.
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', c='k', s=100, linewidth=1)

    # Plot the history of the centroids with lines.
    for j in range(centroids.shape[0]):
        plt.plot([centroids[j, :][0], previous[j, :][0]],
                 [centroids[j, :][1], previous[j, :][1]], c='k')
    # Title
    plt.title('Iteration number {:d}'.format(i+1))

# Create a function to run the K-means algorithm.
def runkMeans(X, initial_centroids, max_iters, plot_progress):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example. It uses initial_centroids used as the initial
    centroids. max_iters specifies the total number of interactions 
    of K-Means to execute. plot_progress is a true/false flag that 
    indicates if the function should also plot its progress as the 
    learning happens. This is set to false by default. runkMeans returns 
    centroids, a K x n matrix of the computed centroids and idx, a m x 1 
    vector of centroid assignments (i.e. each entry in range [1..K])
    Args:
        X                : array(# training examples, 2)
        initial_centroids: array(# of centroids, 2)
        max_iters        : int, # of iterations
        plot_progress    : boolean, default set to False
    Returns:
        centroids        : array(# of centroids, 2)
        idx              : array(# training examples, 1)
    """
    # Initialize values.
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    
    # Run K-Means.
    # Turn interactive mode on to refresh the plot and generate one final plot.
    plt.ion()
    for i in range(max_iters):
        # Output progress.
        print('K-Means iteration {}/{}...'.format(i, max_iters))
        
        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
        
        # Optionally, plot progress here.
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids.
        centroids = computeCentroids(X, idx, K)

    return centroids, idx

# Set K-Means variables.
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice we generate them automatically, such as by
# setting them to be random examples.
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm.
centroids, idx = runkMeans(X, initial_centroids, max_iters, plot_progress=True)
print('\nK-Means Done.')
```

    K-Means iteration 0/10...
    K-Means iteration 1/10...
    K-Means iteration 2/10...
    K-Means iteration 3/10...
    K-Means iteration 4/10...
    K-Means iteration 5/10...
    K-Means iteration 6/10...
    K-Means iteration 7/10...
    K-Means iteration 8/10...
    K-Means iteration 9/10...
    
    K-Means Done.



![Figure 1](/coursera_ml_andrew/K-means 1.png "Figure 1: Progress of K-Means")


## Random Initialization

A good strategy for initializing the centroids is to select random examples from the training set. The code in the function `kMeansInitCentroids` first randomly permutes the indices of the examples. Then, it selects the first K examples based on the random permutation of the indices. This allows the examples to be selected at random without the risk of selecting the same example twice.


```python
# Create a function to initialize centroids.
def kMeansInitCentroids(X, K):
    """
    Initializes K centroids that are to be 
    used in K-Means on the dataset X.
    Args:
        X                : array(# training examples, 2)
        K                : int, # of centroids
    Returns:
        initial_centroids: array(# of centroids, 2)
    """
    # Init centroids.
    centroids = np.zeros((K, X.shape[1]))

    # Randomly reorder the indices of examples.
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids.
    centroids = X[randidx[:K], :]
    
    return centroids

# Set K-Means variables.
K = 3
max_iters = 10

initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means algorithm.
centroids, idx = runkMeans(X, initial_centroids, max_iters, plot_progress=True)
print('\nK-Means Done.')
```

    K-Means iteration 0/10...
    K-Means iteration 1/10...
    K-Means iteration 2/10...
    K-Means iteration 3/10...
    K-Means iteration 4/10...
    K-Means iteration 5/10...
    K-Means iteration 6/10...
    K-Means iteration 7/10...
    K-Means iteration 8/10...
    K-Means iteration 9/10...
    
    K-Means Done.



![Figure 2](/coursera_ml_andrew/K-means 2.png "Figure 2: Progress of K-Means with random initial centroids")


## Image Compression with K-means

K-means will be applied to image compression. In a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding. A sample 24-bit color image contains thousands of colors, which can be reduced to 16 colors.

By making this reduction, it is possible to represent (compress) the photo in an efficient way. Specifically, there is only need to store the RGB values of the 16 selected colors, and for each pixel in the image now it is needed to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities).

The K-means algorithm will be applied to select the 16 colors that will be used to represent the compressed image. Concretely, every pixel will be treated in the original image as a data example and the K-means algorithm will be used to find the 16 colors that best group (cluster) the pixels in the 3-dimensional RGB space. Once the cluster centroids have been computed on the image, then the 16 colors will be used to replace the pixels in the original image.

### K-means on Pixels

The following code first loads the image, and then reshapes it to create a `$m \times 3$` matrix of pixel colors `$\left( \text{where } m = 16384 = 128 \times 128 \right)$`, and calls the K-means function on it.

After finding the top K = 16 colors to represent the image, each pixel position can now be assigned to its closest centroid using the `findClosestCentroids` function. This allows to represent the original image using the centroid assignments of each pixel. Notice that the number of bits that are required to describe the image have been significantly reduced. The original image required 24 bits for each one of the `$128\times128$` pixel locations, resulting in total size of `$128\times128\times24 = 393,216$` bits. The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location. The final number of bits used is therefore `$16\times24 + 128\times128\times4 = 65,920$` bits, which corresponds to compressing the original image by about a factor of 6.

Finally, the effects of the compression can be viewed by reconstructing the image based only on the centroid assignments. Specifically, each pixel location can be replaced with the mean of the centroid assigned to it. Even though the resulting image retains most of the characteristics of the original, we also see some compression artifacts.


```python
from PIL import Image

print('Running K-Means clustering on pixels from an image.')

# Load an image of a bird.
A = Image.open('bird_small.png')
A = np.array(A) # array(128, 128, 3)

# Divide by 255 so that all values are in the range 0-1.
A = A / 255

# Get the size of the image.
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives the dataset matrix X that will be used K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3)

# Run K-Means algorithm on this data.
# Different values of K and max_iters can be tried here.
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids randomly. 
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means.
centroids, idx = runkMeans(X, initial_centroids, max_iters, plot_progress=True)
```

    Running K-Means clustering on pixels from an image.
    K-Means iteration 0/10...
    K-Means iteration 1/10...
    K-Means iteration 2/10...
    K-Means iteration 3/10...
    K-Means iteration 4/10...
    K-Means iteration 5/10...
    K-Means iteration 6/10...
    K-Means iteration 7/10...
    K-Means iteration 8/10...
    K-Means iteration 9/10...



![Figure 3](/coursera_ml_andrew/K-means 3.png "Figure 3: Progress of K-Means on pixels of an image")



```python
print('Applying K-Means to compress an image.')

# Find closest cluster members.
idx = findClosestCentroids(X, centroids)

# Essentially, now the image X is represented as in terms of the indices in idx.
# The image can be recoverd from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value.
X_recovered = centroids[idx,:]

# Reshape the recovered image into proper dimensions.
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(A) 
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title('Compressed, with {} colors.'.format(K))
plt.show()
```

    Applying K-Means to compress an image.



![Figure 4](/coursera_ml_andrew/K-means 4.png "Original and reconstructed image")

