import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import math
import os

# --------------------------------------------------------
# DATA INPUT
# --------------------------------------------------------
# Labels for all data points
labels = ["A1","A2","A3","A4","A5","A6","A7","A8"]

# Dataset (x, y) coordinates
points = np.array([
    [2,10], [2,5], [8,4], [5,8],
    [7,5], [6,4], [1,2], [4,9]
], dtype=float)

# Initial cluster centers (C1, C2, C3)
initial_centers = np.array([
    [2,10],  # C1 from A1
    [5,8],   # C2 from A4
    [1,2]    # C3 from A7
], dtype=float)

# --------------------------------------------------------
# K-MEANS HELPER FUNCTIONS
# --------------------------------------------------------

def euclidean(a, b):
    """
    Compute Euclidean distance between points a and b.
    """
    return math.sqrt(((a - b) ** 2).sum())


def assign_clusters(points, centers):
    """
    Assign each point to the nearest center.
    Returns:
        - array of cluster assignments (0,1,2)
        - matrix of distances for reporting
    """
    clusters = []
    distances_all = []

    for p in points:
        distances = [euclidean(p, c) for c in centers]  # distance to each center
        distances_all.append(distances)
        clusters.append(int(np.argmin(distances)))      # closest center index

    return np.array(clusters), distances_all


def update_centers(points, clusters, k, old_centers):
    """
    Compute new centers by averaging points in each cluster.
    If a cluster has no members, keep the old center.
    """
    new_centers = old_centers.copy()

    for i in range(k):
        members = points[clusters == i]  # points belonging to cluster i
        if len(members) > 0:
            new_centers[i] = members.mean(axis=0)

    return new_centers

# --------------------------------------------------------
# PDF SETUP (FPDF)
# --------------------------------------------------------

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

def add_title(text):
    """Add main title text."""
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, txt=text, ln=1)

def add_subtitle(text):
    """Add section subtitle."""
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, txt=text, ln=1)

def add_text(text):
    """Add a block of regular text."""
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, txt=text)

# --------------------------------------------------------
# MATPLOTLIB FUNCTION TO SAVE GRAPH FOR PDF
# --------------------------------------------------------

def save_kmeans_plot(iteration, points, centers, clusters):
    """
    Draw and save a scatter plot showing:
      - points colored by cluster
      - centers marked with X
    The image is saved temporarily and returned.
    """
    colors = ["red", "blue", "green"]

    plt.figure(figsize=(6, 5))

    # Plot each data point
    for i in range(len(points)):
        plt.scatter(points[i,0], points[i,1], color=colors[clusters[i]], s=80)
        plt.text(points[i,0] + 0.1, points[i,1] + 0.1, labels[i])

    # Plot centers
    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], color=colors[i], marker="X", s=200, edgecolor='black')
        plt.text(c[0] + 0.15, c[1] + 0.15, f"C{i+1}", fontsize=12, fontweight="bold")

    plt.title(f"K-means Iteration {iteration}")
    plt.grid(True)

    filename = f"plot_iter_{iteration}.png"
    plt.savefig(filename, dpi=120)
    plt.close()
    return filename

# --------------------------------------------------------
# MAIN K-MEANS LOOP
# --------------------------------------------------------

centers = initial_centers.copy()
k = centers.shape[0]
iteration = 1

# Introduction page
pdf.add_page()
add_title("K-Means Clustering - Full Step-by-Step Results")
add_text("Dataset points: A1-A8\nInitial centers: C1=A1, C2=A4, C3=A7\n")

while True:
    # Step 1 — Assign points to nearest center
    clusters, distances_all = assign_clusters(points, centers)
    
    # Step 2 — Recompute centers
    new_centers = update_centers(points, clusters, k, centers)

    # New page for this iteration
    pdf.add_page()
    add_title(f"Iteration {iteration}")

    # Show centers before update
    add_subtitle("Centers BEFORE iteration")
    text = ""
    for idx, c in enumerate(centers):
        text += f"C{idx+1}: ({c[0]:.3f}, {c[1]:.3f})\n"
    add_text(text)

    # Show distance table for each point
    add_subtitle("Distance Table")
    pdf.set_font("Arial", size=9)

    # Header row
    pdf.cell(25, 6, "Point", 1)
    pdf.cell(20, 6, "x", 1)
    pdf.cell(20, 6, "y", 1)
    for j in range(k):
        pdf.cell(30, 6, f"d_to_C{j+1}", 1)
    pdf.cell(25, 6, "Cluster", 1, ln=1)

    # Table rows
    for i in range(len(points)):
        pdf.cell(25, 6, labels[i], 1)
        pdf.cell(20, 6, f"{points[i,0]:.2f}", 1)
        pdf.cell(20, 6, f"{points[i,1]:.2f}", 1)
        for d in distances_all[i]:
            pdf.cell(30, 6, f"{d:.3f}", 1)
        pdf.cell(25, 6, str(clusters[i] + 1), 1, ln=1)

    # Show centers after update
    add_subtitle("Centers AFTER iteration")
    text = ""
    for idx, c in enumerate(new_centers):
        member_labels = [labels[i] for i in range(len(points)) if clusters[i] == idx]
        members = ", ".join(member_labels) if member_labels else "None"
        text += f"C{idx+1}: ({c[0]:.3f}, {c[1]:.3f})  Members: {members}\n"
    add_text(text)

    # Add visual plot (scatter image)
    img_file = save_kmeans_plot(iteration, points, centers, clusters)
    pdf.image(img_file, x=10, w=180)
    os.remove(img_file)

    # Check if algorithm has converged
    if np.allclose(centers, new_centers):
        pdf.add_page()
        add_title("Convergence Reached")
        add_text(f"K-means converged at iteration {iteration}.")
        break

    # Prepare next iteration
    centers = new_centers.copy()
    iteration += 1

# --------------------------------------------------------
# FINAL SUMMARY PAGE
# --------------------------------------------------------

pdf.add_page()
add_title("Final Summary")

summary = ""
for idx, c in enumerate(centers):
    member_labels = [labels[i] for i in range(len(points)) if clusters[i] == idx]
    summary += f"C{idx+1}: ({c[0]:.3f}, {c[1]:.3f})  Members: {', '.join(member_labels)}\n"

add_text(summary)

# Save output on pdf
pdf.output("kmeans_full_guide_fpdf.pdf")
print("PDF created: kmeans_full_guide_fpdf.pdf")
