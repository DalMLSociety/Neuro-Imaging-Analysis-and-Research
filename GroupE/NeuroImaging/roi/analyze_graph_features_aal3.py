import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =======================
# Load correlation matrices (AAL3 atlas)
# =======================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
result_dir = os.path.join(base_dir, "results", "roi_analysis")
plot_dir = os.path.join(base_dir, "plots", "roi_graphs_aal3")
os.makedirs(plot_dir, exist_ok=True)

# Load matrices
control_matrix = np.load(os.path.join(result_dir, "connectivity_matrix_control_aal3.npy"))
patient_matrix = np.load(os.path.join(result_dir, "connectivity_matrix_patient_aal3.npy"))

# =======================
# Compute graph features per group
# =======================
def analyze_group(matrix, group, threshold=0.3):
    # Thresholding (remove weak edges)
    matrix[np.abs(matrix) < threshold] = 0

    # Build graph
    G = nx.from_numpy_array(matrix)

    # Compute graph metrics
    global_conn = np.mean(np.abs(matrix[matrix != 0]))
    density = nx.density(G)
    avg_degree = np.mean([d for _, d in G.degree()])
    modularity = None
    n_communities = None

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        n_communities = len(communities)
        modularity = nx.algorithms.community.modularity(G, communities)
    except Exception as e:
        print(f"Modularity failed: {e}")

    # Save graph image
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color='lightcoral', edge_color='gray',
            with_labels=False, node_size=100, alpha=0.85, width=0.8)
    plt.title(f"Connectivity Graph - {group.capitalize()} Group (AAL3)")
    fig_path = os.path.join(plot_dir, f"connectivity_graph_{group}_aal3.png")
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
features_path = os.path.join(base_dir, "results", "graph_features_aal3.csv")
df.to_csv(features_path, index=False)
print(f"\nGraph features saved to: {features_path}")
