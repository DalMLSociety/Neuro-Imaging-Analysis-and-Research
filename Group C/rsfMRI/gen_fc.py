# functional_connectivity.py

import os
import nibabel as nib
import numpy as np
from nilearn import datasets, input_data, connectome
from util import get_mri_file_path

# === Step 1: Load 4D functional MRI image ===
file_path = get_mri_file_path(root_name='MRIData',
                              path=['sub-kaneff01', 'func', 'sub-kaneff01_task-effloc_run-001_bold.nii.gz'])
fmri_img = nib.load(file_path)

# === Step 2: Load brain parcellation atlas (AAL) ===
atlas = datasets.fetch_atlas_aal()
atlas_filename = atlas.maps
region_labels = atlas.labels

# === Step 3: Extract average time series for each brain region ===
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename,
                                       standardize=True,
                                       detrend=True,
                                       memory='nilearn_cache')
time_series = masker.fit_transform(fmri_img)  # shape: (n_timepoints, n_regions)

# === Step 4: Compute correlation matrix ===
conn_measure = connectome.ConnectivityMeasure(kind='correlation')
correlation_matrix = conn_measure.fit_transform([time_series])[0]

# === Step 5: Save results ===
np.save('connectivity_matrix_run001.npy', correlation_matrix)
print("Connectivity matrix shape:", correlation_matrix.shape)
