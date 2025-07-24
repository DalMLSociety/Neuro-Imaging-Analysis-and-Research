# File: 5_2_sara_ind_people.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
from Util.util_io import mri_path_niar
from Util.config import NIAR, dmn_coords_33
import matplotlib.cm as cm

# configure masker for 33 ROIs
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
    compute Pearson correlation matrix, and return the upper-triangle vector.
    Returns array of length D = 33*32/2 = 528.
    """
    img = nib.load(mri_path_niar(NIAR, sub_id, year))
    ts = masker.fit_transform(img)        # shape = (timepoints, 33)
    mat = np.corrcoef(ts.T)               # shape = (33, 33)
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]

def main():
    # 1. Read clinical data
    df = pd.read_excel('demoSCA7.xlsx')
    # assign subject IDs 1…16 and year labels 1…3
    df['subject'] = df.index // 3 + 1
    df['year']    = df.groupby('subject').cumcount() + 1

    # 2. Sort all 48 samples by their SARA score (pure across years)
    df_sorted = df.sort_values('sara').reset_index(drop=True)

    # 3. Build identifiers for x‑tick labels: e.g. 'P01y02(14.5)'
    labels = []
    for _, row in df_sorted.iterrows():
        p = f"P{int(row['subject']):02d}"
        y = f"y{int(row['year']):02d}"
        s = f"({row['sara']})"
        labels.append(f"{p}{y}{s}")

    # 4. Compute FC vectors for each sorted sample
    fc_list = []
    for iddx, row in df_sorted.iterrows():
        print("start", iddx)
        sub_id = f"P{int(row['subject']):02d}"
        year   = str(int(row['year']))
        fc_vec = compute_static_fc(sub_id, year)
        fc_list.append(fc_vec)
    fc_mat = np.vstack(fc_list)  # shape = (48, 528)

    # 5. Plot each ROI‑pair time series across sorted samples, with distinct colors
    n_samples, D = fc_mat.shape
    x = np.arange(n_samples)
    cmap = cm.get_cmap('viridis', D)

    plt.figure(figsize=(12, 6))
    for j in range(D):
        plt.plot(x, fc_mat[:, j], color=cmap(j), alpha=0.6, linewidth=0.8)
    plt.xlabel('Samples (sorted by SARA)')
    plt.ylabel('ROI–ROI correlation')
    plt.title('Static FC for all ROI‑pairs across 48 samples\n(sorted by increasing SARA)')
    plt.xticks(x, labels, rotation=90, fontsize='small')
    plt.tight_layout()
    plt.savefig('roi_pair_fc_all_samples.png', dpi=150)
    plt.close()

    print(f"Plotted {D} ROI‑pair lines over {n_samples} samples.")
    print("Output saved to 'roi_pair_fc_all_samples.png'")

if __name__ == '__main__':
    main()
