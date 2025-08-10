import numpy as np
import os

# Parameters
window_size = 30  # in TRs
step_size = 5  # in TRs
output_dir = "C:/Users/msset/MRI/output/dfc_matrices/"
os.makedirs(output_dir, exist_ok=True)

# Path to time series (output of Step 1)
input_dir = "C:/Users/msset/MRI/output/time_series_aal3/"  # adjust if different


def compute_dfc(time_series, window_size=30, step_size=5):
    n_timepoints, n_rois = time_series.shape
    dfc_vectors = []

    for start in range(0, n_timepoints - window_size + 1, step_size):
        window_ts = time_series[start:start + window_size]
        corr_mat = np.corrcoef(window_ts.T)

        iu = np.triu_indices(n_rois, k=1)
        dfc_vec = corr_mat[iu]
        dfc_vectors.append(dfc_vec)

    return np.array(dfc_vectors)  # shape: (n_windows, n_features)


# Process each time series file
for file in os.listdir(input_dir):
    if file.endswith(".npy"):
        subject_id = file.replace(".npy", "")
        ts = np.load(os.path.join(input_dir, file))  # shape: (T, ROIs)

        print(f"Computing dFC for {subject_id} with shape {ts.shape}")
        dfc = compute_dfc(ts, window_size=window_size, step_size=step_size)
        print(f" → Output shape: {dfc.shape} (windows × features)")

        # Save result
        np.save(os.path.join(output_dir, f"{subject_id}_dfc.npy"), dfc)
