import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT
from datetime import datetime

# Setup
s_id = "p03"
year = "1"
img = nib.load(mri_path_niar(NIAR, s_id, year))
data = img.get_fdata()
z_slice = 30
output_dir = os.path.join(OUTPUT_ROOT, format_output_name(f"bold_z{z_slice}"))
os.makedirs(output_dir, exist_ok=True)

# Loop over time
n_timepoints = data.shape[3]
for t in range(n_timepoints):
    slice_2d = data[:, :, z_slice, t].T  # transpose for correct orientation

    vmin, vmax = np.percentile(slice_2d, [1, 99])

    plt.imshow(slice_2d, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    plt.axis("off")
    plt.title(f"BOLD z={z_slice}, t={t}")
    filename = os.path.join(output_dir, f"bold_z{z_slice}_t{t:03d}.png")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()

print("All frames saved.")
