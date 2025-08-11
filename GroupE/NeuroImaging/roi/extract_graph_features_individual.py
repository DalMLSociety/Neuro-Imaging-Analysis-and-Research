import os
import pandas as pd
import numpy as np
import networkx as nx

# ============== Paths ==============
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_dir = os.path.join(base_dir, "data", "roi_time_series")
output_file = os.path.join(base_dir, "results", "graph_features_per_subject_schaefer.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ============== Parameters ==============
atlas_filter = "schaefer"
groups = ["control", "patient"]
threshold = 0.3

# ============== Main Loop ==============
features = []

for group in groups:
    group_path = os.path.join(data_dir, group)
    for filename in os.listdir(group_path):
        if not filename.endswith(".csv") or atlas_filter not in filename.lower():
            continue

        try:
            # Parse subject and session info from filename
            parts = filename.replace(".csv", "").split("_")
            subject_id = parts[1]  # e.g., C01 or P03
            session_id = parts[2]  # e.g., rs-1

            # Load time series and compute correlation matrix
            filepath = os.path.join(group_path, filename)
            ts = pd.read_csv(filepath, header=0)
            corr_matrix = ts.corr().values

            # Threshold the matrix
            corr_matrix[np.abs(corr_matrix) < threshold] = 0

            # Create graph
            G = nx.from_numpy_array(corr_matrix)

            # Extract graph features
            global_conn = np.mean(np.abs(corr_matrix[corr_matrix != 0]))
            density = nx.density(G)
            avg_degree = np.mean([deg for _, deg in G.degree()])
            modularity = None
            num_communities = None

            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G))
                num_communities = len(communities)
                modularity = nx.algorithms.community.modularity(G, communities)
            except Exception as e:
                print(f"[Warning] Modularity failed for {filename}: {e}")

            features.append({
                "subject": subject_id,
                "session": session_id,
                "group": group,
                "atlas": "schaefer",
                "global_connectivity": global_conn,
                "graph_density": density,
                "average_degree": avg_degree,
                "modularity": modularity,
                "n_communities": num_communities
            })

        except Exception as e:
            print(f"[Error] Failed processing {filename}: {e}")

# ============== Save Output ==============
df = pd.DataFrame(features)
df.to_csv(output_file, index=False)
print(f"\n Saved individual graph features to: {output_file}")
