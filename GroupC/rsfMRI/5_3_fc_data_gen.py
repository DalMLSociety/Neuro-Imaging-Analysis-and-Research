# File: 5_3_fc_data_gen.py

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
from Util.util_io import mri_path_niar
from Util.config import OUTPUT_ROOT, NIAR, dmn_coords_33

# configure masker for the 33 ROIs
masker = NiftiSpheresMasker(
    seeds=dmn_coords_33,
    radius=5.0,
    detrend=True,
    standardize=True,
    t_r=2.0
)

def compute_static_fc(sub_id: str, year: str) -> np.ndarray:
    """
    Load subject's fMRI for given year, extract ROI time series,
    compute Pearson correlation matrix, and return the upper-triangle vector.
    """
    # build the path with the correct signature
    fmri_path = mri_path_niar(NIAR, sub_id, year)
    img = nib.load(fmri_path)
    ts  = masker.fit_transform(img)    # shape = (timepoints, 33)
    mat = np.corrcoef(ts.T)            # shape = (33, 33)
    iu  = np.triu_indices_from(mat, k=1)
    return mat[iu]                     # length = 33*32/2 = 528

def main():
    # 1) read clinical spreadsheet
    df = pd.read_excel('demoSCA7.xlsx')

    # 2) add subject (1–16) and year (1–3) columns
    df['subject'] = df.index // 3 + 1
    df['year']    = df.groupby('subject').cumcount() + 1

    records = []
    # 3) iterate through all 48 samples
    for _, row in df.iterrows():
        subj       = int(row['subject'])
        yr         = int(row['year'])
        age        = row['age']
        year_onset = row['yearOnset']
        cag        = row['cag']
        school     = row['school']
        sara       = row['sara']

        sub_id = f"P{subj:02d}"
        year   = str(yr)

        # compute FC vector
        fc_vec = compute_static_fc(sub_id, year)

        # assemble record
        rec = {
            'subject':    subj,
            'year':       yr,
            'age':        age,
            'yearOnset':  year_onset,
            'cag':        cag,
            'school':     school,
            'sara':       sara
        }
        for i, val in enumerate(fc_vec, start=1):
            rec[f'fc_{i}'] = val

        records.append(rec)
        print(f"→ Computed FC for {sub_id} year{year}: SARA={sara}")

    # 4) build DataFrame
    out_df = pd.DataFrame.from_records(records)

    # 5) save to disk
    out_dir = os.path.join(OUTPUT_ROOT, 'fc_datasets')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'fc_data_16x3.csv')
    npz_path = os.path.join(out_dir, 'fc_data_16x3.npz')

    out_df.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        subject   = out_df['subject'].values,
        year      = out_df['year'].values,
        age       = out_df['age'].values,
        yearOnset = out_df['yearOnset'].values,
        cag       = out_df['cag'].values,
        school    = out_df['school'].values,
        sara      = out_df['sara'].values,
        fc        = out_df.filter(like='fc_').values
    )

    print(f"\nDataset saved to:\n  {csv_path}\n  {npz_path}")

if __name__ == '__main__':
    main()
