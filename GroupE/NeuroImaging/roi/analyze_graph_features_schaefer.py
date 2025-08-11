import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =======================
# Load correlation matrices (Schaefer atlas)
# =======================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
result_dir = os.path.join(base_dir, "results", "roi_analysis")
plot_dir = os.path.join(base_dir, "plots", "roi_graphs")
os.makedirs(plot_dir, exist_ok=True)

# Load matrices
control_matrix = np.load(os.path.join(result_dir, "connectivity_matrix_control_schaefer.npy"))
patient_matrix = np.load(os.path.join(result_dir, "connectivity_matrix_patient_schaefer.npy"))

# =======================
# Compute graph features per group
# =======================
def analyze_group(matrix, group, threshold=0.3):
    # Thresholding (remove weak edges)
    matrix[np.abs(matrix) < threshold] = 0

    # Construct a graph from the correlation matrix
    G = nx.from_numpy_array(matrix)

    # Calculate features
    global_conn = np.mean(np.abs(matrix[matrix != 0]))
    density = nx.density(G)
    avg_degree = np.mean([d for n, d in G.degree()])
    modularity = None
    n_communities = None

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        n_communities = len(communities)
        membership = {}
        for i, comm in enumerate(communities):
            for node in comm:
                membership[node] = i
        modularity = nx.algorithms.community.modularity(G, communities)
    except Exception as e:
        print(f"Modularity failed: {e}")

    # Save chart
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        node_color='deepskyblue',
        edge_color='gray',
        with_labels=False,
        node_size=120,
        alpha=0.8,
        width=0.8
    )
    plt.title(f"Connectivity Graph - {group.capitalize()} Group (Threshold = {threshold})")
    fig_path = os.path.join(plot_dir, f"connectivity_graph_{group}_schaefer.png")
    plt.savefig(fig_path)
    plt.close()

    print(f"Graph for {group} saved to: {fig_path}")

    return {
        "Group": group,
        "Global Connectivity": global_conn,
        "Graph Density": density,
        "Average Degree": avg_degree,
        "Modularity": modularity,
        "Number of Communities": n_communities
    }

# =======================
# Run for both groups and save features
# =======================
control_features = analyze_group(control_matrix, "control")
patient_features = analyze_group(patient_matrix, "patient")

# Save to CSV
df = pd.DataFrame([control_features, patient_features])
features_path = os.path.join(base_dir, "results", "graph_features_schaefer.csv")
df.to_csv(features_path, index=False)
print(f"\nGraph features saved to: {features_path}")
