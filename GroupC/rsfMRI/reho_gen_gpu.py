import os
import numpy as np
import nibabel as nib
import cupy as cp
from scipy.stats import rankdata
from tqdm import tqdm
from Util.util_io import get_mri_file_path
from Util.config import OUTPUT_ROOT

# Load data
bold_path = get_mri_file_path(dataset_name="MRIData",
                              path=["sub-kaneff01", "func", "sub-kaneff01_task-effloc_run-001_bold.nii.gz"])
output_path = os.path.join(OUTPUT_ROOT, "reho", "reho_map_gpu.nii.gz")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

bold_img = nib.load(bold_path)
bold_data = bold_img.get_fdata()
X, Y, Z, T = bold_data.shape

# Kernel offsets for 3x3x3 neighborhood
kernel = [(i, j, k) for i in [-1, 0, 1]
                      for j in [-1, 0, 1]
                      for k in [-1, 0, 1]]

reho_map = np.zeros((X, Y, Z))
pbar = tqdm(total=(X-2)*(Y-2)*(Z-2), desc="ReHo on GPU", ncols=100)

# Compute ReHo
for x in range(1, X-1):
    for y in range(1, Y-1):
        for z in range(1, Z-1):
            neighbors = []
            for dx, dy, dz in kernel:
                ts = bold_data[x+dx, y+dy, z+dz, :]
                if np.any(np.isnan(ts)):
                    break
                neighbors.append(ts)
            else:
                neighbors_np = np.array(neighbors)  # (27, T)
                ranks_np = np.apply_along_axis(rankdata, 0, neighbors_np)  # rank along each timepoint
                ranks_cp = cp.asarray(ranks_np)

                rank_sum = cp.sum(ranks_cp, axis=0)
                S = cp.sum((rank_sum - 27 * (T + 1) / 2) ** 2)
                W = 12 * S / (27**2 * (T**3 - T))
                reho_map[x, y, z] = float(W.get())  # back to CPU
            pbar.update(1)

pbar.close()

# Save result
reho_img = nib.Nifti1Image(reho_map, affine=bold_img.affine)
nib.save(reho_img, output_path)
print(f"ReHo map saved to {output_path}")
