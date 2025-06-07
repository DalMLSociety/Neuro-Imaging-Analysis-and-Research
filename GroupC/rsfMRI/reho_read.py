import os
import nibabel as nib
import numpy as np
from nilearn import plotting
from Util.util_io import mri_path
from Util.config import OUTPUT_ROOT, NIAR

# Paths
reho_path = os.path.join(OUTPUT_ROOT, "reho", "reho_map.nii.gz")
anat_path = mri_path(
    name=NIAR,
    path=["sub-kaneff01", "anat", "sub-kaneff01_T1w.nii.gz"]
)
save_path = os.path.join(OUTPUT_ROOT, "reho", "reho_vis.png")

# Load images
reho_img = nib.load(reho_path)
anat_img = nib.load(anat_path)

# Normalize values to [0, 1]
data = reho_img.get_fdata()
if np.max(data) > 0:
    data /= np.max(data)
    reho_img = nib.Nifti1Image(data, affine=reho_img.affine)

# Determine display range
threshold = 0.2  # you can tune this
vmax = 1.0

# Plot
display = plotting.plot_stat_map(
    reho_img,
    bg_img=anat_img,
    display_mode="ortho",
    cut_coords=(0, -16, 20),
    threshold=threshold,
    vmax=vmax,
    cmap="hot",
    title="ReHo Map"
)

display.savefig(save_path, dpi=300)
display.close()
print(f"Saved to: {save_path}")
