import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd

from nilearn.input_data import NiftiSpheresMasker
from nilearn.plotting import plot_matrix

from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT, dmn_coords_33, dmn_names_33

# === Global Layout ===
plt.rcParams['figure.constrained_layout.use'] = True

# === Parameter Settings ===
s_id = "C01"
year = "1"
func_path = mri_path_niar(NIAR, s_id, year)
img = nib.load(func_path)


# === Extract ROI Mean Time Series ===
masker = NiftiSpheresMasker(
    seeds=dmn_coords_33,
    radius=6.0,            # sphere radius 6 mm
    detrend=True,
    standardize=True,
    t_r=2.0                # adjust according to actual TR
)
ts = masker.fit_transform(img)  # (n_timepoints, 33)

# === Compute Functional Connectivity Matrix (Pearson r) ===
fc_matrix = np.corrcoef(ts.T)   # shape = (33, 33)

# === Prepare Output Directory ===
output_dir = os.path.join(
    OUTPUT_ROOT,
    format_output_name(f"fc_DMN_{s_id}_year{year}")
)
os.makedirs(output_dir, exist_ok=True)

# === 1) Plot and Save Static FC Heatmap (Original -1 to 1 Red-White-Blue) ===
# … The part of extracting ts and computing fc_matrix remains unchanged …

fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)

# use imshow + nearest + vmin=0, vmax=1
im = ax.imshow(
    fc_matrix,
    cmap="RdBu_r",
    vmin=-1.0, vmax=1.0,           # <— only show [-1, 1] range
    interpolation='nearest',
    aspect='equal'
)

# Labels
ax.set_xticks(np.arange(len(dmn_names_33)))
ax.set_yticks(np.arange(len(dmn_names_33)))
ax.set_xticklabels(dmn_names_33, rotation=90, fontsize=6)
ax.set_yticklabels(dmn_names_33, fontsize=6)
ax.set_title(f"Static FC of DMN (−1 to 1 range)\nsubj {s_id}, year {year}")

# colorbar also from -1 to 1
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
cbar.set_label('Pearson r')

# Save
fig.savefig(os.path.join(output_dir, "dmn_static_fc_0to1.png"), dpi=300)
plt.close(fig)


# === 2) Export Edge List to CSV ===
idx, jdx = np.triu_indices_from(fc_matrix, k=1)
df = pd.DataFrame({
    "ROI1": [dmn_names_33[i] for i in idx],
    "ROI2": [dmn_names_33[j] for j in jdx],
    "r":    fc_matrix[idx, jdx]
})
csv_path = os.path.join(output_dir, "dmn_static_fc_edges.csv")
df.to_csv(csv_path, index=False)
