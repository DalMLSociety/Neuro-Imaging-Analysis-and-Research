import os
import matplotlib.pyplot as plt
from nilearn import input_data, plotting
import numpy as np
import nibabel as nib
from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT, seed_points
import matplotlib
matplotlib.use('pdf')

# ---- Config ----
s_id = "C02"
seed_name = "PCC"
radius = 8
z_slices = [-20, -4, 0, 12, 24, 40, 52, 64]

# ---- Output ----
output_path = os.path.join(OUTPUT_ROOT, format_output_name(f"seed-based_grid_{s_id}_{seed_name}.png"))

# ---- Load years ----
years = ["1", "2", "3"]
fc_maps = []

for year in years:
    img = nib.load(mri_path_niar(NIAR, s_id, year))

    seed_masker = input_data.NiftiSpheresMasker(
        [seed_points[seed_name]], radius=radius, detrend=True, standardize=True)
    seed_ts = seed_masker.fit_transform(img)

    brain_masker = input_data.NiftiMasker(standardize=True)
    brain_ts = brain_masker.fit_transform(img)

    fc_map = np.dot(brain_ts.T, seed_ts) / brain_ts.shape[0]
    corr_img = brain_masker.inverse_transform(fc_map.T)
    fc_maps.append(corr_img)

# ---- Create figure ----
fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(20, 7))
fig.suptitle(f"Seed-based FC: {s_id}, seed={seed_name}", fontsize=16)

for row, (year, fc_img) in enumerate(zip(years, fc_maps)):
    for col, z in enumerate(z_slices):
        ax = axes[row, col]
        display = plotting.plot_stat_map(
            fc_img,
            threshold=0.2,
            display_mode="z",
            cut_coords=[z],
            colorbar=False,
            axes=ax,
            annotate=False,
            title=f"z={z}" if row == 0 else None
        )
        if col == 0:
            ax.set_ylabel(f"Year {year}", fontsize=12)

# ---- Save ----
plt.subplots_adjust(wspace=0.05, hspace=0.15, top=0.9)
plt.subplots_adjust(top=0.9)  # 留空间给 title
plt.savefig(output_path, dpi=300)
print(f"[Saved] {output_path}")
