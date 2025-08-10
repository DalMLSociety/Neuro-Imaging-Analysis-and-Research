import os
import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_aal
from nilearn.input_data import NiftiLabelsMasker

# Load AAL3 atlas
aal3 = fetch_atlas_aal(version="3v2")
atlas_img = aal3.maps
atlas_labels = aal3.labels  # can print region names

# data and output folder
fmri_dir = "C:/Users/msset/MRI/raw_data"  # replace with your .nii folder path
output_dir = "C:/Users/msset/MRI/output/time_series_aal3/"  # where you want to save .npy files
os.makedirs(output_dir, exist_ok=True)

# Setup masker to extract mean signal from each atlas region
masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True,
    detrend=True,
    t_r=2.0,  # change if your TR is different
    verbose=1
)

# Loop over each subject’s .nii file
for file in os.listdir(fmri_dir):
    if file.endswith(".nii") or file.endswith(".nii.gz"):
        subject_id = file.replace(".nii", "").replace(".nii.gz", "")
        print(f"Processing {subject_id}")

        fmri_path = os.path.join(fmri_dir, file)
        fmri_img = nib.load(fmri_path)

        # Extract time series: shape = (n_timepoints, n_regions)
        time_series = masker.fit_transform(fmri_img)
        print(f" → Time series shape: {time_series.shape}")

        # Save time series as .npy
        out_path = os.path.join(output_dir, f"{subject_id}_timeseries.npy")
        np.save(out_path, time_series)
        print(f"Saved to: {out_path}")
