import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT

# === Parameter Settings ===
s_id = "C01"
year = "1"
func_path = mri_path_niar(NIAR, s_id, year)
img = nib.load(func_path)

# === 33 DMN ROI Coordinates & Names ===
dmn_coords = [
    (-11,  55,  -5), ( 11,  53,  -6),
    (-10,  50,  20), ( 10,  50,  19),
    (-20,  31,  46), ( 23,  32,  46),
    ( -5, -50,  35), (  7, -51,  34),
    ( -6, -55,  12), (  6, -54,  13),
    (-46, -64,  33), ( 50, -59,  34),
    (-58, -21, -15), ( 59, -17, -18),
    (-38,  17, -34), ( 43,  15, -35),
    (-36,  23, -16), ( 37,  25, -16),
    (-24, -30, -16), ( 26, -26, -18),
    (-15,  -9, -18), ( 17,  -8, -16),
    (-11,  12,   7), ( 13,  11,   9),
    (-26, -82, -33), ( 29, -79, -34),
    ( -6, -57, -45), (  8, -53, -48),
    ( -7, -14,   8), (  7, -11,   8),
    ( -7,  12, -12), (  7,   9, -12),
    (  0, -22, -21)
]
dmn_names = [
    "R VMPFC", "R AMPFC", "R DLPFC", "R PCC", "R Rsp", "R PH", "R Amy",
    "R VLPFC", "R TP",   "R MTG",   "R PPC", "R T",   "R BF",  "R C",
    "R CbH",   "R CbT",  "MidB",
    "L VMPFC", "L AMPFC","L DLPFC", "L PCC", "L Rsp", "L PH", "L Amy",
    "L VLPFC","L TP",   "L MTG",   "L PPC", "L T",   "L BF",  "L C",
    "L CbH",   "L CbT"
]

# === Keep 32 ROIs: Drop MidB ===
drop = dmn_names.index("MidB")
coords_32 = [c for i, c in enumerate(dmn_coords) if i != drop]
names_32  = [n for i, n in enumerate(dmn_names ) if i != drop]
n_rois = len(names_32)    # should be 32

# === Extract Time Series ===
masker = NiftiSpheresMasker(
    seeds=coords_32,
    radius=6.0,
    detrend=True,
    standardize=True,
    t_r=2.0
)
ts = masker.fit_transform(img)  # shape = (n_timepoints, 32)
n_tp = ts.shape[0]

# === Compute Weight Matrix and Self-Prediction Correlation using SVR ===
W = np.zeros((n_rois, n_rois))
corr_self = np.zeros(n_rois)

for i in range(n_rois):
    y = ts[:, i]
    X = np.delete(ts, i, axis=1)
    Xs = StandardScaler().fit_transform(X)

    svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
    svr.fit(Xs, y)
    w = svr.coef_.ravel()

    # Fill back into the weight matrix
    W[i, :i]    = w[:i]
    W[i, i+1:]  = w[i:]

    # Self-prediction correlation coefficient
    y_pred = svr.predict(Xs)
    corr_self[i] = np.corrcoef(y, y_pred)[0,1]

# === Output Directory ===
output_dir = os.path.join(
    OUTPUT_ROOT,
    format_output_name(f"mfc_DMN32_{s_id}_year{year}")
)
os.makedirs(output_dir, exist_ok=True)

# === Visualize Weight Matrix ===
import matplotlib
matplotlib.use('Agg')
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(
    W, cmap='RdBu_r',
    vmin=-np.max(np.abs(W)),
    vmax= np.max(np.abs(W))
)
ax.set_xticks(np.arange(n_rois))
ax.set_xticklabels(names_32, rotation=90, fontsize=6)
ax.set_yticks(np.arange(n_rois))
ax.set_yticklabels(names_32, fontsize=6)
ax.set_title(f"SVR-based MFC (32 ROI)\n {s_id}, year {year}")
fig.colorbar(im, ax=ax, label='Weight')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "mfc32_weight_matrix.png"), dpi=300)
plt.close(fig)

# === Visualize Self-Prediction Correlation Distribution ===
fig, ax = plt.subplots(figsize=(5,3))
ax.hist(corr_self, bins=10)
ax.set_xlabel('Self‐prediction r')
ax.set_ylabel('Count')
ax.set_title('SVR Self‐prediction Accuracy')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "mfc32_self_pred_hist.png"), dpi=300)
plt.close(fig)

print(f"32‐ROI SVR-based MFC outputs saved in:\n  {output_dir}")
