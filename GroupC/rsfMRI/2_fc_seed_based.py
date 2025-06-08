import os
from nilearn import input_data, plotting
import numpy as np
import nibabel as nib
from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT, seed_points
import matplotlib
matplotlib.use('MacOSX')


# input path
s_id = "C01"
r_id = "3"
img = nib.load(mri_path_niar(NIAR, s_id, r_id))

# output path
output_dir = os.path.join(OUTPUT_ROOT, format_output_name('seed-based'))
os.makedirs(output_dir, exist_ok=True)


# Extract time series from PCC seed region
seed_masker = input_data.NiftiSpheresMasker(
    [seed_points["L_IPL"]],
    radius=8,
    detrend=True,
    standardize=True)
seed_ts = seed_masker.fit_transform(img)

# Extract time series from the whole brain
brain_masker = input_data.NiftiMasker(standardize=True)
brain_ts = brain_masker.fit_transform(img)

# FC map (Pearson correlation)
fc_map = np.dot(brain_ts.T, seed_ts) / brain_ts.shape[0]
corr_img = brain_masker.inverse_transform(fc_map.T)


z_min, z_max, tick = -72, 110, 2
cut_coords = list(range(z_min, z_max, tick))


# start slice
for idx, z in enumerate(cut_coords, start=1):
    display = plotting.plot_stat_map(
        corr_img,
        threshold=0.2,
        display_mode="z",
        cut_coords=[z]
    )

    filename = f"fc_seed-based_{idx:03d}.png"
    output_path = os.path.join(output_dir, filename)
    display.savefig(output_path, dpi=300)
    display.close()
    print(f"[Saved] {output_path}")

print("All slices saved.")
