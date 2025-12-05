import numpy as np
from fpdf import FPDF

# Define data points
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

# Initial centroids
centroids = np.array([
    [2, 10],  # C1
    [5, 8],   # C2
    [1, 2]    # C3
])

# Functions for K-means
def assign_clusters(points, centroids):
    clusters = []
    for point in points:
        distances = np.linalg.norm(point - centroids, axis=1)
        cluster_id = np.argmin(distances)
        clusters.append(cluster_id)
    return np.array(clusters)

def update_centroids(points, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = points[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            new_centroid = centroids[i]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# Prepare PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="K-means Clustering Output", ln=True, align='C')
pdf.ln(5)

# Run K-means iterations
k = 3
for iteration in range(2):
    clusters = assign_clusters(points, centroids)
    centroids = update_centroids(points, clusters, k)
    
    pdf.cell(200, 8, txt=f"Iteration {iteration + 1}:", ln=True)
    pdf.cell(200, 8, txt=f"Centroids: {centroids}", ln=True)
    pdf.cell(200, 8, txt=f"Cluster assignments: {clusters}", ln=True)
    pdf.ln(5)

# Save PDF
pdf.output("kmeans_output.pdf")
print("PDF saved as 'kmeans_output.pdf'")
