import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd

from nilearn.input_data import NiftiSpheresMasker
from nilearn.plotting import plot_matrix

from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT

# === Global Layout ===
plt.rcParams['figure.constrained_layout.use'] = True

# === Parameter Settings ===
s_id = "C01"
year = "1"
func_path = mri_path_niar(NIAR, s_id, year)
img = nib.load(func_path)

# MNI center coordinates for 33 DMN ROIs
dmn_coords = [
    (-11,  55,  -5), ( 11,  53,  -6),
    (-10,  50,  20), ( 10,  50,  19),
    (-20,  31,  46), ( 23,  32,  46),
    ( -5, -50,  35), (  7, -51,  34),
    ( -6, -55,  12), (  6, -54,  13),
    (-46, -64,  33), ( 50, -59,  34),
    (-58, -21, -15), ( 59, -17, -18),
    (-38,  17, -34), ( 43,  15, -35),
    (-36,  23, -16), ( 37,  25, -16),
    (-24, -30, -16), ( 26, -26, -18),
    (-15,  -9, -18), ( 17,  -8, -16),
    (-11,  12,   7), ( 13,  11,   9),
    (-26, -82, -33), ( 29, -79, -34),
    ( -6, -57, -45), (  8, -53, -48),
    ( -7, -14,   8), (  7, -11,   8),
    ( -7,  12, -12), (  7,   9, -12),
    (  0, -22, -21)
]

# **Naming 33 ROIs in the order given in the paper's 'Functional space' figure**
dmn_names = [
    # Right hemisphere
    "R VMPFC", "R AMPFC", "R DLPFC", "R PCC", "R Rsp", "R PH", "R Amy",
    "R VLPFC","R TP",   "R MTG",   "R PPC", "R T",   "R BF",  "R C",
    "R CbH",   "R CbT",  "MidB",
    # Left hemisphere
    "L VMPFC", "L AMPFC","L DLPFC", "L PCC", "L Rsp", "L PH", "L Amy",
    "L VLPFC","L TP",   "L MTG",   "L PPC", "L T",   "L BF",  "L C",
    "L CbH",   "L CbT"
]

# === Extract ROI Mean Time Series ===
masker = NiftiSpheresMasker(
    seeds=dmn_coords,
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
ax.set_xticks(np.arange(len(dmn_names)))
ax.set_yticks(np.arange(len(dmn_names)))
ax.set_xticklabels(dmn_names, rotation=90, fontsize=6)
ax.set_yticklabels(dmn_names,           fontsize=6)
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
    "ROI1": [dmn_names[i] for i in idx],
    "ROI2": [dmn_names[j] for j in jdx],
    "r":    fc_matrix[idx, jdx]
})
csv_path = os.path.join(output_dir, "dmn_static_fc_edges.csv")
df.to_csv(csv_path, index=False)
