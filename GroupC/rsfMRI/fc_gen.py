import os
import nibabel as nib
import numpy as np
from nilearn import datasets, input_data, connectome
from Util.util_io import mri_path_niar
from Util.config import OUTPUT_ROOT, NIAR

# load 4D functional MRI image
s_id = "C01"
r_id = "3"
img = nib.load(mri_path_niar(NIAR, "C01", "3"))

# Load brain parcellation atlas (AAL)
atlas = datasets.fetch_atlas_aal()
atlas_filename = atlas.maps
region_labels = atlas.labels

# Extract average time series for each brain region
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename,
                                       standardize=True,
                                       detrend=True,
                                       memory='nilearn_cache')
time_series = masker.fit_transform(img)  # shape: (n_timepoints, n_regions)

# Compute correlation matrix
conn_measure = connectome.ConnectivityMeasure(kind='correlation')
correlation_matrix = conn_measure.fit_transform([time_series])[0]

save_path = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001.npy')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, correlation_matrix)
print("Connectivity matrix shape:", correlation_matrix.shape)
