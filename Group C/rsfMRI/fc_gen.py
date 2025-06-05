import os
import nibabel as nib
import numpy as np
from nilearn import datasets, input_data, connectome
from Util.util import get_mri_file_path
from Util.config import OUTPUT_ROOT

# load 4D functional MRI image
file_path = get_mri_file_path(dataset_name='MRIData',
                              path=['sub-kaneff01', 'func', 'sub-kaneff01_task-effloc_run-001_bold.nii.gz'])
img = nib.load(file_path)

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
