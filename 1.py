# make_kmeans_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import numpy as np
import math

# Data and labels
labels = ["A1","A2","A3","A4","A5","A6","A7","A8"]
points = np.array([
    [2, 10],   # A1
    [2, 5],    # A2
    [8, 4],    # A3
    [5, 8],    # A4
    [7, 5],    # A5
    [6, 4],    # A6
    [1, 2],    # A7
    [4, 9]     # A8
], dtype=float)

initial_centers = np.array([
    [2, 10],  # C1 = A1
    [5, 8],   # C2 = A4
    [1, 2]    # C3 = A7
], dtype=float)

def euclidean(a, b):
    return math.sqrt(((a-b)**2).sum())

# PDF setup
file_path = "kmeans_full_steps.pdf"
doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
styles = getSampleStyleSheet()
styleN = styles["Normal"]
styleH = styles["Heading1"]
styleH2 = styles["Heading2"]
styleSmall = ParagraphStyle('small', parent=styles['Normal'], fontSize=9)

elements = []
elements.append(Paragraph("K-Means Clustering — Full Step-by-Step Tables", styleH))
elements.append(Spacer(1, 8))
elements.append(Paragraph("Dataset: A1(2,10), A2(2,5), A3(8,4), A4(5,8), A5(7,5), A6(6,4), A7(1,2), A8(4,9)", styleN))
elements.append(Paragraph("Initial centers: C1=A1(2,10), C2=A4(5,8), C3=A7(1,2)", styleN))
elements.append(Spacer(1, 12))

centers = initial_centers.copy()
k = centers.shape[0]
max_iters = 20

for it in range(1, max_iters+1):
    # compute distances
    distances = [[euclidean(points[i], centers[j]) for j in range(k)] for i in range(points.shape[0])]
    assigned = [min(range(k), key=lambda j: distances[i][j]) for i in range(len(distances))]
    # compute new centers
    new_centers = centers.copy()
    for j in range(k):
        members = [points[i] for i in range(len(points)) if assigned[i]==j]
        if len(members)>0:
            new_centers[j] = np.mean(members, axis=0)
    # Build table for iteration
    elements.append(Paragraph(f"Iteration {it}", styleH2))
    # centers before
    cb_data = [["Center","cx","cy"]]
    for idx,c in enumerate(centers):
        cb_data.append([f"C{idx+1}", f"{c[0]:.6f}", f"{c[1]:.6f}"])
    tcb = Table(cb_data, hAlign='LEFT')
    tcb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.25,colors.black)]))
    elements.append(Paragraph("Centers before this iteration:", styleSmall))
    elements.append(tcb)
    elements.append(Spacer(1,6))
    # distances table header
    header = ["Point","x","y"] + [f"d_to_C{j+1}" for j in range(k)] + ["AssignedCluster"]
    data = [header]
    for i,label in enumerate(labels):
        row = [label, f"{points[i,0]:.6f}", f"{points[i,1]:.6f}"]
        for j in range(k):
            row.append(f"{distances[i][j]:.6f}")
        row.append(str(assigned[i]+1))
        data.append(row)
    # adjust column widths as needed
    colWidths = [40,50,50] + [70]*k + [80]
    t = Table(data, repeatRows=1, colWidths=colWidths)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.25,colors.black),
                           ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('FONTSIZE',(0,0),(-1,-1),8)]))
    elements.append(t)
    elements.append(Spacer(1,6))
    # centers after
    ca_data = [["Center","cx","cy","Members (labels)"]]
    for j in range(k):
        members_labels = [labels[i] for i in range(len(points)) if assigned[i]==j]
        ca_data.append([f"C{j+1}", f"{new_centers[j,0]:.6f}", f"{new_centers[j,1]:.6f}", ", ".join(members_labels) if members_labels else "None"])
    tca = Table(ca_data, hAlign='LEFT', colWidths=[50,70,70,230])
    tca.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.25,colors.black),
                             ('FONTSIZE',(0,0),(-1,-1),8)]))
    elements.append(Paragraph("Centers after this iteration:", styleSmall))
    elements.append(tca)
    elements.append(Spacer(1,12))
    # check convergence
    if np.allclose(centers, new_centers):
        elements.append(Paragraph(f"Converged at iteration {it} (centers unchanged).", styles["Normal"]))
        break
    centers = new_centers.copy()
    # page break every 2 iterations to keep readability
    if it % 2 == 0:
        elements.append(PageBreak())

# Final summary page
elements.append(PageBreak())
elements.append(Paragraph("Final Summary", styleH2))
final_centers_data = [["Center","cx","cy","Members (labels)"]]
for j in range(k):
    members_labels = [labels[i] for i in range(len(points)) if assigned[i]==j]
    final_centers_data.append([f"C{j+1}", f"{centers[j,0]:.6f}", f"{centers[j,1]:.6f}", ", ".join(members_labels)])
tf = Table(final_centers_data, colWidths=[50,70,70,230])
tf.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.25,colors.black),('FONTSIZE',(0,0),(-1,-1),9)]))
elements.append(tf)

doc.build(elements)
print("PDF created:", file_path)
