import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from nilearn import datasets
from Util.config import OUTPUT_ROOT


# convert FC matrix to Graph
def matrix_to_graph(matrix, labels, threshold=0.7):
    G = nx.Graph()
    n = matrix.shape[0]
    for i in range(n):
        G.add_node(labels[i])
        for j in range(i + 1, n):
            w = matrix[i, j]
            if abs(w) >= threshold:
                G.add_edge(labels[i], labels[j], weight=w)
    return G


# load matrix and label
matrix_path = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001.npy')
matrix = np.load(matrix_path)
atlas = datasets.fetch_atlas_aal()
labels = atlas.labels

# build graph
G = matrix_to_graph(matrix, labels, threshold=0.7)

# analyze structure
deg = nx.degree_centrality(G)
top_nodes = sorted(deg.items(), key=lambda x: -x[1])[:10]

# save top10 centrality
output_txt = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001_top_nodes.txt')
with open(output_txt, 'w') as f:
    for name, val in top_nodes:
        f.write(f"{name}: {val:.3f}\n")

# visualize
plt.figure(figsize=(12, 12))
nx.draw_spring(G, with_labels=True, node_size=150, font_size=6, edge_color='gray')
fig_path = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001_graph.png')
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"Graph structure saved to: {fig_path}")
print(f"Top central nodes saved to: {output_txt}")
