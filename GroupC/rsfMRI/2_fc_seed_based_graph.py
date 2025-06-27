import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import networkx as nx
from nilearn import input_data, plotting
from matplotlib import cm, colors
from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT, seed_points

s_id = "p16"
radius = 8
corr_threshold = 0.2
years = ["1", "2", "3"]

selected_seeds = ["PCC", "mPFC", "L_IPL", "R_IPL", "ACC", "L_Insula", "R_Insula",
                  "DLPFC_L", "DLPFC_R", "L_IPS", "R_IPS", "L_FEF", "R_FEF"]
region_coords = [seed_points[name] for name in selected_seeds]
region_labels = selected_seeds

def process_year(year: str):
    print(f"[INFO] Processing subject {s_id}, year {year}...")

    masker = input_data.NiftiSpheresMasker(
        seeds=region_coords, radius=radius,
        detrend=True, standardize=True
    )
    img = nib.load(mri_path_niar(NIAR, s_id, year))
    ts = masker.fit_transform(img)

    corr_matrix = np.corrcoef(ts.T)

    G = nx.Graph()
    for i, label_i in enumerate(region_labels):
        for j, label_j in enumerate(region_labels):
            if i < j:
                weight = corr_matrix[i, j]
                if abs(weight) > corr_threshold:
                    G.add_edge(label_i, label_j, weight=weight)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    norm = colors.Normalize(vmin=min(weights), vmax=max(weights))
    edge_colors = cm.viridis(norm(weights))

    nx.draw(
        G, pos, ax=ax, with_labels=True,
        node_color='skyblue', edge_color=edge_colors,
        width=2.5, font_size=10
    )

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Pearson correlation (r)")
    ax.set_title(f"Functional Connectivity Graph: {s_id}, Year {year}")
    plt.tight_layout()

    graph_path = os.path.join(OUTPUT_ROOT, format_output_name(f"graph_fc_{s_id}_{year}.png"))
    plt.savefig(graph_path, dpi=300)
    plt.close()
    print(f"[Saved] {graph_path}")

    display = plotting.plot_connectome(
        corr_matrix, region_coords,
        title=f"Connectome View: {s_id}, Year {year}",
        node_size=40,
        edge_threshold=corr_threshold
    )
    connectome_path = os.path.join(OUTPUT_ROOT, format_output_name(f"connectome_fc_{s_id}_{year}.png"))
    display.savefig(connectome_path, dpi=300)
    display.close()
    print(f"[Saved] {connectome_path}")


for y in years:
    process_year(y)
