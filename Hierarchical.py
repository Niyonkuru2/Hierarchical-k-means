# Hierarchical Clustering Example in Python

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# ---------------------------------------------------------
# Step 1: Define the data points
# ---------------------------------------------------------
# Each row is a 2D coordinate (x, y)
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

labels = ['A1','A2','A3','A4','A5','A6','A7','A8']

# ---------------------------------------------------------
# Step 2: Compute the linkage matrix
# ---------------------------------------------------------
# Ward method minimizes variance in clusters
Z = linkage(points, method='ward')

# ---------------------------------------------------------
# Step 3: Plot the dendrogram
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
dendrogram(Z, labels=labels, leaf_rotation=45)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# Step 4: Cut the dendrogram to form clusters
# ---------------------------------------------------------
max_clusters = 3
clusters = fcluster(Z, max_clusters, criterion='maxclust')

# ---------------------------------------------------------
# Step 5: Print cluster assignment
# ---------------------------------------------------------
print("Cluster assignments:")
for i, cluster_id in enumerate(clusters):
    print(f"Point {labels[i]} -> Cluster {cluster_id}")

# ---------------------------------------------------------
# Step 6: 2D scatter plot of clusters
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(points[:, 0], points[:, 1], c=clusters)

# Add labels on each point
for i, label in enumerate(labels):
    plt.text(points[i,0] + 0.1, points[i,1] + 0.1, label)

plt.title("2D Cluster Visualization")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid(True)
plt.show()
