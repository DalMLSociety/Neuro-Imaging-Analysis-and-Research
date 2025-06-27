import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask, compute_epi_mask
from sklearn.cluster import KMeans
from nilearn.image import new_img_like
from nilearn.plotting import plot_roi
from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT

# ----- Config -----
s_id = "C01"
year = "3"
n_clusters = 10
z_slices_to_plot = [10, 20, 30, 40]


output_dir = os.path.join(OUTPUT_ROOT, format_output_name("kmeans_clusters"))
os.makedirs(output_dir, exist_ok=True)

# ----- Load image -----
img = nib.load(mri_path_niar(NIAR, s_id, year))
mask_img = compute_epi_mask(img)

# ----- Extract time series from brain voxels -----
data_matrix = apply_mask(img, mask_img).T  # (voxels, timepoints)

# ----- Perform K-means clustering -----
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(data_matrix)

# ----- Map labels back to 3D volume -----
mask_data = mask_img.get_fdata().astype(bool)
cluster_data = np.zeros(mask_data.shape)
cluster_data[mask_data] = labels
cluster_img = new_img_like(mask_img, cluster_data)

# ----- Save cluster image -----
cluster_img.to_filename(os.path.join(output_dir, "kmeans_cluster_labels.nii.gz"))

# ----- Save visualizations -----
for z in z_slices_to_plot:
    display = plot_roi(cluster_img,
                       display_mode="z",
                       cut_coords=[z],
                       cmap="tab10")
    fname = os.path.join(output_dir, f"cluster_z{z:02d}.png")
    display.savefig(fname, dpi=300)
    display.close()

print(f"K-means clustering done. Results saved to: {output_dir}")
