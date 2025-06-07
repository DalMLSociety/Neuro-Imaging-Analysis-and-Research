import numpy as np
import nibabel as nib
import os
from scipy.stats import rankdata
from nilearn.masking import compute_brain_mask
from tqdm import tqdm
from Util.util_io import mri_path_niar
from Util.config import OUTPUT_ROOT, NIAR

# Load data
s_id = "C01"
r_id = "3"
bold_img = nib.load(mri_path_niar(NIAR, "C01", "3"))

output_path = os.path.join(OUTPUT_ROOT, "reho", "reho_map.nii.gz")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

bold_data = bold_img.get_fdata()
X, Y, Z, T = bold_data.shape

# Compute brain mask
mask_img = compute_brain_mask(bold_img)
mask_data = mask_img.get_fdata().astype(bool)

# Define kernel
kernel = [(i, j, k) for i in [-1, 0, 1]
                      for j in [-1, 0, 1]
                      for k in [-1, 0, 1]]
n = len(kernel)

# ReHo map initialization
reho_map = np.zeros((X, Y, Z))
pbar = tqdm(total=(X-2)*(Y-2)*(Z-2), desc="Computing ReHo", ncols=100)

# Compute ReHo (correct Kendall's W)
for x in range(1, X-1):
    for y in range(1, Y-1):
        for z in range(1, Z-1):
            if not mask_data[x, y, z]:
                pbar.update(1)
                continue

            neighbors = []
            for dx, dy, dz in kernel:
                nx, ny, nz = x+dx, y+dy, z+dz
                if not mask_data[nx, ny, nz]:
                    break
                ts = bold_data[nx, ny, nz, :]
                if np.any(np.isnan(ts)):
                    break
                neighbors.append(ts)
            else:
                R = np.apply_along_axis(rankdata, 0, np.array(neighbors))  # shape: (n, T)
                S = np.sum((np.sum(R, axis=0) - n*(T+1)/2)**2)
                W = 12 * S / (n**2 * (T**3 - T))  # standard Kendall's W
                W /= T  # normalization
                reho_map[x, y, z] = W

            pbar.update(1)

pbar.close()

# Save result
reho_img = nib.Nifti1Image(reho_map, affine=bold_img.affine)
nib.save(reho_img, output_path)
print(f"ReHo map saved to: {output_path}")
