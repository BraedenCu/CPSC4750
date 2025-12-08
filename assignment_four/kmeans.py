from __future__ import print_function
from math import inf
import numpy as np
import scipy
import matplotlib.pyplot as plt

def kmeans(X, k, init=None):
    """ KMEANS implements the k-means algorithm.

    [CLUSTERS, CENTROIDS] = KMEANS(X, K) partitions the data points
    in the N-by-P data matrix X into K distinct clusters, using Euclidean
    distance. This is a simple implementation of the k-means algorithm with
    random initialization.

    Optionally, it takes the argument INIT, a K-by-P matrix with a fixed
    initial position for the cluster centroids.  

    MYKMEANS returns an N-by-1 vector CLUSTERS containing the cluster
    indices of each data point, as well as CENTROIDS, a K-by-P matrix with 
    the final cluster centroids' locations.
    """


    n, p = X.shape

    if init is None:
      #choose initial centroids by picking k points at random from X
      init = X[np.random.randint(n, size=k), :]
    
    #centroids is a k-by-p random matrix
    #its i^th row contains the coordinates of the cluster with index i
    centroids = init

    #initialize cluster assignment array
    clusters = np.zeros(n)

    MAXITER = 1000

    for iter in range(MAXITER):
        
        #create a new clusters vector to fill in with updated assignments
        new_clusters = np.zeros(n)
        

        #for each data point x_i
        for i in range(n):
            
            x_i = X[i,:]
            
            #find closest cluster
            closest = findClosestCluster(x_i,centroids)###IMPLEMENT THIS FUNCTION AT THE END OF THIS FILE

            #reassign x_i to the index of the closest centroid found
            new_clusters[i] = closest

        
        
        if hasConverged(clusters,new_clusters):###IMPLEMENT THIS FUNCTION AT THE END OF THIS FILE
            #exit loop
            break 
        
        
        #otherwise, update assignment
        clusters = new_clusters
        #and recompute centroids
        centroids = recomputeCentroids(X,clusters,k)###IMPLEMENT THIS FUNCTION AT THE END OF THIS FILE

    

    if iter == (MAXITER-1):
        print('Maximum number of iterations reached!')
    
    return clusters, centroids
    

def findClosestCluster(x_i,centroids):
    # Compute Euclidean distance from x_i to each cluster centroid and return 
    # the index of the closest one (an integer).
    # NOTE: use of numpy/scipy.linalg.norm function is NOT allowed here.

    ### Replace the following line with your own code
    
    # we have p different data points

    curr_min = inf
    closest = 0

    for c in range(centroids.shape[0]):
        curr_dist = 0
        for i in range(centroids.shape[1]):
            curr_dist += ((centroids[c][i] - x_i[i]) ** 2)
            
        if curr_dist < curr_min:
            curr_min = curr_dist
            closest = c


    return closest



def hasConverged(old_assignment, new_assignment):
    # Check if algorithm has converged, i.e., cluster assignments haven't
    # changed since last iteration. Return a boolean.

    ### Replace the following line with your own code
   # converged = True;

    return np.array_equal(old_assignment, new_assignment)



def recomputeCentroids(X,clusters,k):
    # Recompute centroids based on current cluster assignment.
    # Return a k-by-p array where each row is a centroid. 
    n, p = X.shape
    ### Replace the following line with your own code

    cluster_list = [[] for _ in range(k)] # a list of arrays, where cluster_list[i] contains points in X who are in cluster i

    for i in range(len(clusters)):
        cluster_list[int(clusters[i])].append(X[i])
    
    centroids = []

    for i in range(len(cluster_list)):
        if len(cluster_list[i]) > 0:
            centroids.append(np.mean(np.array(cluster_list[i]), axis=0))
        else:
            centroids.append(np.zeros(p))


    return np.array(centroids)



# QUESTION 2

# 2a
data = scipy.io.loadmat('logos.mat')['data']

n_images = 10
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i in range(10):
    row = i // 5
    col = i % 5

    img = data[:, :, i]
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].axis('off')

plt.show()

# Yes, I can tell that the first row of Yale's have a much more distinguished letter Y's compared to the second row.


# 2b

# need to flatten each of the images into a 4800 element array and then stack them

X = []

for i in range(10):
    img = data[:, :, i]
    flattened = img.flatten()

    X.append(flattened)

# Convert to numpy array: shape (10, 4800)
X = np.array(X)

# given that we have 4800 pixels, we will have a dimension of 4800

# It would be rather difficult to show plot these images since it would be a 4800
# dimension space, and it is very difficult to plot 4800 dimensions. If we wanted to
# we could use PCA to plot the top 3 dimenesions.


# 2c
# Run k-means to split the data into 2 different clusters

out_clusters, out_centroids = kmeans(X, 2)

for i in range(len(out_clusters)):
    print(f"Cluster {i}: {out_clusters[i]}")


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i in range(2):
    centroid_img = out_centroids[i, :].reshape(48, 100)
    axes[i].imshow(centroid_img, cmap='gray')
    axes[i].set_title(f'Centroid {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# We know that k-means depends on the initial randomization.
# Therefore, each image may go to a different cluster and 
# each cluster may have a different local minima. 

'''
(base) Rajs-MacBook-Pro-2:assn4 rajjadhav$ python3 kmeans.py
Cluster 0: 0.0
Cluster 1: 0.0
Cluster 2: 0.0
Cluster 3: 0.0
Cluster 4: 1.0
Cluster 5: 0.0
Cluster 6: 0.0
Cluster 7: 0.0
Cluster 8: 0.0
Cluster 9: 0.0
(base) Rajs-MacBook-Pro-2:assn4 rajjadhav$ python3 kmeans.py
Cluster 0: 0.0
Cluster 1: 1.0
Cluster 2: 0.0
Cluster 3: 0.0
Cluster 4: 1.0
Cluster 5: 1.0
Cluster 6: 1.0
Cluster 7: 1.0
Cluster 8: 1.0
Cluster 9: 0.0
(base) Rajs-MacBook-Pro-2:assn4 rajjadhav$ 
'''

# QUESTION 3 PCA

# 3A


def pca(X, m):

    # step 1
    row_mean = np.mean(X, axis=1)

    for i in range(X.shape[0]):
        X[i] -= row_mean[i]

    # step 2

    C = (1 / (X.shape[0]-1)) * X.T @ X

    # step 3

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(C, k=m)
    

    # runtime warning told me to do this (check not too sure)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    sort_index = np.argsort(eigenvalues)[::-1] 
    
    eigenvalues_sorted = eigenvalues[sort_index]
    eigenvectors_sorted = eigenvectors[:, sort_index]  
    
    eigenvalues_top = eigenvalues_sorted[:m]
    eigenvectors_top = eigenvectors_sorted[:, :m]

    return eigenvalues_top, eigenvectors_top


# 3b

G = scipy.io.loadmat('gaussian.mat')['gaussian']

g_eigenvalues, g_eigenvectors = pca(G, 2)

# Get the mean of the data for centering the quiver plot
# Need column mean (mean of each feature) to get 2D position [mean_x, mean_y]
g_col_mean = np.mean(G, axis=0)

# Scale principal components by their eigenvalues (variances)
# eigenvectors[:, 0] is first PC, eigenvectors[:, 1] is second PC
PC1_scaled = g_eigenvalues[0] * g_eigenvectors[:, 0]
PC2_scaled = g_eigenvalues[1] * g_eigenvectors[:, 1]

# Create scatter plot of the data points
plt.figure(figsize=(10, 10))
plt.scatter(G[:, 0], G[:, 1], alpha=0.5, s=10)

plt.quiver([g_col_mean[0], g_col_mean[0]], 
           [g_col_mean[1], g_col_mean[1]], 
           [np.sqrt(g_eigenvalues[0]) * g_eigenvectors[0, 0], np.sqrt(g_eigenvalues[1]) * g_eigenvectors[0, 1]], 
           [np.sqrt(g_eigenvalues[0]) * g_eigenvectors[1, 0], np.sqrt(g_eigenvalues[1]) * g_eigenvectors[1, 1]], 
           angles='xy', scale_units='xy', scale=1, 
           color=['red', 'blue'], width=0.003, label=['PC1', 'PC2'])

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('PCA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# QUESTION 4

# 4a

logos_data = scipy.io.loadmat('logos.mat')['data']


logos_X = []
for i in range(logos_data.shape[2]):
    img = logos_data[:, :, i]
    flattened = img.flatten()
    logos_X.append(flattened)


logos_X = np.array(logos_X)  

logos_eigenvalues, logos_eigenvectors = pca(logos_X, 4)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for i in range(4):
    row = i // 2
    col = i % 2
    
    pc_img = logos_eigenvectors[:, i].reshape(48, 100)
    axes[row, col].imshow(pc_img, cmap='gray')
    axes[row, col].set_title(f'PC {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(range(1, 5), logos_eigenvalues)
plt.xlabel('Principal Component')
plt.ylabel('Variance (Eigenvalue)')
plt.title('Variances of First 4 Principal Components')
plt.xticks(range(1, 5))
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# 4c

logos_X_centered = logos_X.copy()
row_mean = np.mean(logos_X_centered, axis=1)
for i in range(logos_X_centered.shape[0]):

    logos_X_centered[i] -= row_mean[i]

# Project onto first principal component
first_principal_comp = logos_eigenvectors[:, 0]  # Shape: (4800,)
logos_X_projected = logos_X_centered @ first_principal_comp  # Shape: (10,)

# First we had 10 images, each with 4800 pixels, so 10x4800 shape
# Then we should have an output of 10x1 (dimension being 1D) because we are outputting 

# The components matrix (eigenvectors) give us a new basis where each column is a basis in the 
# original 4800d space. When we project, we actually transform the 4800d space to a 10d space
# The purpose of the 1st PC is that it shows the most variance, making it the most importatn direction. 

# 4d
logos_X_projected = logos_X_projected.reshape(-1, 1)

reduced_clusters, reduced_centroids = kmeans(logos_X_projected, 2)

for i in range(len(reduced_clusters)):
    print(f"Cluster {i}: {reduced_clusters[i]}")

plt.figure(figsize=(10, 6))
colors = ['red' if c == 0 else 'blue' for c in reduced_clusters]
plt.scatter(logos_X_projected, np.zeros(len(logos_X_projected)), c=colors, s=100, alpha=0.7)
plt.xlabel('First principal component')
plt.ylabel('')
plt.title('k-means clustering (reduced data) (4d)')
plt.yticks([])
plt.grid(True, alpha=0.3, axis='x')
plt.show()

centroid_images = []

for cluster_id in range(2):

    cluster_images = []
    for j in range(len(reduced_clusters)):
        if reduced_clusters[j] == cluster_id:
            cluster_images.append(logos_X[j])
    

    centroid = np.mean(cluster_images, axis=0).reshape(48, 100)
    centroid_images.append(centroid)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i in range(2):
    axes[i].imshow(centroid_images[i], cmap='gray')
    axes[i].set_title(f'Centroid {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 4e
# Compare results from part 2 (full data) with part 4d (reduced data)
# The cluster assignments from part 2c (out_clusters) and part 4d (reduced_clusters) 
# are mostly likely going to be different (rather than the same) due to our random initialization.

# For part 2, we use a coordinate system. We see that the original pixel coordinate system (4800 dimensions).
# Each pixel is a dimension, so we have 4800 basis vectors (each one being what I believe itself is some linear combination of the RGB of the pixel.)
# In our coordinate system, we are able to use the Euclidean distance between the pixels as a difference in the brightness.

# In 4d where we used reduced data however, we used a principal component coordinate system  in one dimension. 
# Here our basis vector is just the first principal component which, as mentioned earlier, captures the most amount 
# of variance making it the most important direction. Each image would just be a scalar, a scalar that represents how 
# much that image aligns in that principal component. Our distance in this coordinate system
# is just the differencee between the basis vector and the projection.

# Part 4d (Reduced Data):
# - We used the principal component coordinate system (1 dimension)
# - The basis vector is the first principal component (captures most variance)
# - Each image is now just a scalar value (projection onto PC1)
# - Euclidean distance in 1D is just the absolute difference between values

# QUESTION 5

# 5a
# Load faces data
faces_data = scipy.io.loadmat('faces.mat')['Data']

faces_data = faces_data.astype(np.float64) # not sure if needed tbh

# Normalize each row so max value is 1
for i in range(faces_data.shape[0]):
    row_max = np.max(faces_data[i])
    if row_max > 0:
        faces_data[i] = faces_data[i] / row_max

# Display some sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    row = i // 5
    col = i % 5
    img = faces_data[i, :].reshape(96, 96).T
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Run PCA to get first 9 principal components (eigenfaces)
faces_eigenvalues, faces_eigenvectors = pca(faces_data.copy(), 9)

# Display eigenfaces in 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for i in range(9):
    row = i // 3
    col = i % 3
    eigenface = faces_eigenvectors[:, i].reshape(96, 96).T
    axes[row, col].imshow(eigenface, cmap='gray')
    axes[row, col].set_title(f'Eigenface {i+1}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# 5b
faces_data_normalized = faces_data.copy()
row_mean = np.mean(faces_data_normalized, axis=1)
for i in range(faces_data_normalized.shape[0]):
    faces_data_normalized[i] -= row_mean[i]

U, s, Vt = np.linalg.svd(faces_data_normalized, full_matrices=False)
V = Vt.T  

n = faces_data_normalized.shape[0]
v = (s**2) / (n - 1)

print(f"PCA eigencvals {faces_eigenvalues.shape}")
print(f"SVD variances {v[:9].shape}")
print(f"\nPCA eigenvalues \n{faces_eigenvalues}")
print(f"\nSVD variances  \n{v[:9]}")


eigenvalue_diff = np.abs(faces_eigenvalues - v[:9])
max_diff = np.max(eigenvalue_diff)
if max_diff < 0.01:  # arbitrary check what it should be 
    print("max_diff < 0.01")
else:
    print("max_diff >= 0.01 ")


print(f"PCA eigenvectors : {faces_eigenvectors.shape}")
print(f"SVD V matrix (first 9 cols) shape: {V[:, :9].shape}")

eigenvector_match = True
for i in range(9):
    dot_product = np.dot(faces_eigenvectors[:, i], V[:, i])

    if abs(dot_product - 1) < 0.01 or abs(dot_product + 1) < 0.01:
        continue
    else:
        eigenvector_match = False
        break

# Eigenvectors match (allowing sign differences): should be True
# This makes sense because PCA and SVD are mathematically related - they should give the same results
# The V matrix from SVD contains the same eigenvectors as PCA (up to a sign change)
# and the variances v = s^2 / (n-1) match the PCA eigenvalues