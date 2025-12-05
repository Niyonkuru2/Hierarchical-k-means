# Hierarchical Clustering Example in Python

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Step 1: Define the data points
points = np.array([
    [2, 10],  # A1
    [2, 5],   # A2
    [8, 4],   # A3
    [5, 8],   # A4
    [7, 5],   # A5
    [6, 4],   # A6
    [1, 2],   # A7
    [4, 9]    # A8
])

# Step 2: Compute the linkage matrix
# method='ward' minimizes the variance of clusters being merged
Z = linkage(points, method='ward')

# Step 3: Plot the dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z, labels=['A1','A2','A3','A4','A5','A6','A7','A8'], leaf_rotation=45)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

# Step 4: Form flat clusters (e.g., 3 clusters)
max_clusters = 3
clusters = fcluster(Z, max_clusters, criterion='maxclust')

# Step 5: Print cluster assignment
for i, cluster_id in enumerate(clusters):
    print(f"Point A{i+1} -> Cluster {cluster_id}")
