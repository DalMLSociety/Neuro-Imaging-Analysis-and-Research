import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
import os
from tqdm import tqdm

from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT, dmn_coords_33, dmn_names_33

# ROI masker (already defined)
masker = NiftiSpheresMasker(
    seeds=dmn_coords_33,
    radius=5.0,
    detrend=True,
    standardize=True,
    t_r=2.0
)

def compute_static_fc(sub_id: str, year: str) -> np.ndarray:
    """
    Load the subject's fMRI for the given year, extract time series from 33 ROIs,
    compute the static FC, and return the upper-triangle vector.
    Returns: shape = (D,), where D = 33*32/2 = 528
    """
    # 1) Load the fMRI file
    path = mri_path_niar(NIAR, sub_id, year)
    img = nib.load(path)
    # 2) Extract ROI time series & compute correlation matrix
    ts = masker.fit_transform(img)      # (timepoints, 33)
    mat = np.corrcoef(ts.T)              # (33, 33)
    # 3) Vectorize the upper triangle (excluding diagonal)
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]

def build_subject_features(sub_id: str, years=("1", "2", "3")) -> np.ndarray:
    """
    For one subject, get FC vectors for three years, compute deltas, mean,
    standard deviation, and slope, then concatenate into a single feature vector of length 8D.
    """
    # 1) Read the three years of FC; fc_list is a list of three arrays of shape (D,)
    fc_list = [compute_static_fc(sub_id, y) for y in years]
    fc_arr  = np.vstack(fc_list)          # shape = (3, D)

    # 2) Compute delta features
    delta_21 = fc_arr[1] - fc_arr[0]      # year2 minus year1
    delta_32 = fc_arr[2] - fc_arr[1]      # year3 minus year2

    # 3) Compute statistics: mean, standard deviation, slope
    mean_feat = np.mean(fc_arr, axis=0)
    std_feat = np.std(fc_arr, axis=0)
    t_pts = np.array([1, 2, 3])
    # Linear least squares: design matrix for slope calculation
    A = np.vstack([t_pts, np.ones_like(t_pts)]).T  # (3,2)
    solution = np.linalg.lstsq(A, fc_arr, rcond=None)[0]  # shape = (2, D)
    slope_f = solution[0, :]                               # shape = (D,)

    # 4) Concatenate: 3D + 2D + 3D = 8D features
    return np.concatenate([
        fc_arr.flatten(),  # FC_year1, FC_year2, FC_year3 → 3D
        delta_21,          # D
        delta_32,          # D
        mean_feat,         # D
        std_feat,          # D
        slope_f            # D
    ])


if __name__ == '__main__':
    # Define subject lists
    control_ids = [f"C{str(i).zfill(2)}" for i in range(1, 17)]
    missing = {"C03", "C09", "C10", "C13"}  # Subjects with missing data
    # Remove missing controls (now 12 controls vs. 16 patients)
    control_ids = [sub for sub in control_ids if sub not in missing]
    patient_ids = [f"P{str(i).zfill(2)}" for i in range(1, 17)]
    all_ids = control_ids + patient_ids

    # Cache directory for feature vectors
    cache_dir = os.path.join(OUTPUT_ROOT, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # compute and save cache
    for sub in tqdm(all_ids, desc="Computing features"):
        cache_path = os.path.join(cache_dir, f"{sub}_features.npy")
        if not os.path.exists(cache_path):
            feats = build_subject_features(sub)
            np.save(cache_path, feats)

    # load cached features to assemble X, y
    X_list, y_list = [], []
    for sub in all_ids:
        feats = np.load(os.path.join(cache_dir, f"{sub}_features.npy"))
        label = 0 if sub in control_ids else 1
        X_list.append(feats)
        y_list.append(label)

    X = np.vstack(X_list)  # shape = (28, 8*528 = 4224)
    y = np.array(y_list)   # shape = (28,)

    print("Loaded X shape:", X.shape)
    print("Loaded y shape:", y.shape)

    # — Save X and y locally —
    data_dir = os.path.join(OUTPUT_ROOT, "dataset")
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(data_dir, "features_labels.npz"),
        X=X,
        y=y
    )
    print("Saved X,y to", os.path.join(data_dir, "features_labels.npz"))
