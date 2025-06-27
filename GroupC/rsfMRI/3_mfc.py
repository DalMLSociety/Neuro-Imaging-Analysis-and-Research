import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT

# === Parameter Settings ===
s_id = "C01"
year = "1"
atlas_name = 'cort-maxprob-thr25-2mm'

# 1. Load preprocessed fMRI (4D NIfTI)
func_path = mri_path_niar(NIAR, s_id, year)
img = nib.load(func_path)

# 2. Load atlas and remove background
atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
labels_img  = atlas.maps
label_names = atlas.labels[1:]      # skip 'Background'

# 3. Extract time series for all ROIs
masker = NiftiLabelsMasker(
    labels_img,
    standardize=True,
    detrend=True,
    t_r=2.0,
    low_pass=0.1,
    high_pass=0.01
)
time_series = masker.fit_transform(img)  # (n_tp, n_rois)
n_tp, n_rois = time_series.shape

# 4. Specify DMN core node names (adjust according to atlas.labels)
dmn_labels = [
    "Precuneous Cortex",
    "Cingulate Gyrus, posterior division",
    "Cingulate Gyrus, anterior division",
    "Frontal Pole",
    "Superior Frontal Gyrus, medial division",
    "Angular Gyrus",
    "Lateral Occipital Cortex, superior division"
]

# 5. Find their indices in label_names
dmn_idx = [i for i, name in enumerate(label_names) if name in dmn_labels]
names_dmn = [label_names[i] for i in dmn_idx]
n_dmn = len(dmn_idx)

# 6. Subset the whole-brain time series to DMN nodes
ts_dmn = time_series[:, dmn_idx]    # (n_tp, n_dmn)

# 7. Train SVR on DMN subset and build weight matrix
W_dmn = np.zeros((n_dmn, n_dmn))
corr_pred_dmn = np.zeros(n_dmn)

for ii in range(n_dmn):
    y = ts_dmn[:, ii]
    X = np.delete(ts_dmn, ii, axis=1)
    Xs = StandardScaler().fit_transform(X)

    svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
    svr.fit(Xs, y)
    w = svr.coef_.ravel()

    W_dmn[ii, :ii]   = w[:ii]
    W_dmn[ii, ii+1:] = w[ii:]

    y_pred = svr.predict(Xs)
    corr_pred_dmn[ii] = np.corrcoef(y, y_pred)[0,1]

# 8. Prepare output directory
output_dir = os.path.join(
    OUTPUT_ROOT,
    format_output_name(f"mfc_DMN_{s_id}_year{year}")
)
os.makedirs(output_dir, exist_ok=True)

# 9. Visualize and save DMN weight matrix
import matplotlib
matplotlib.use('Agg')
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(W_dmn,
               cmap='RdBu_r',
               vmin=-np.max(np.abs(W_dmn)),
               vmax=np.max(np.abs(W_dmn)))
ax.set_xticks(np.arange(n_dmn))
ax.set_xticklabels(names_dmn, rotation=45, ha='right')
ax.set_yticks(np.arange(n_dmn))
ax.set_yticklabels(names_dmn)
ax.set_title(f"DMN MFC weights: subj {s_id}, year {year}")
fig.colorbar(im, ax=ax, label='Weight')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "dmn_mfc_weight_matrix.png"), dpi=300)
plt.close(fig)

# 10. Visualize and save self-prediction correlation histogram
fig, ax = plt.subplots(figsize=(5,3))
ax.hist(corr_pred_dmn, bins=10)
ax.set_xlabel('Self-prediction corr.')
ax.set_ylabel('Count')
ax.set_title('DMN within-run accuracy')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "dmn_prediction_accuracy_hist.png"), dpi=300)
plt.close(fig)

print(f"DMN MFC outputs saved to: {output_dir}")
