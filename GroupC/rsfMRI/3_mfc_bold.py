import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT

# === Parameter Settings ===
s_id = "C01"
year = "1"

# 1. Load preprocessed fMRI (4D NIfTI)
func_path = mri_path_niar(NIAR, s_id, year)
img = nib.load(func_path)
data = img.get_fdata()  # shape = (nx, ny, nz, nt)
nx, ny, nz, nt = data.shape

# 2. Merge spatial dimensions into a single voxel dimension
data_2d = data.reshape(-1, nt).T   # shape = (nt, n_voxels)
n_vox = data_2d.shape[1]

# 3. Select an example voxel (you can change to any linear index you’re interested in)
voxel_idx = 123456  # <— change to the voxel index you want to examine (0 ~ n_vox-1)

# 4. Construct regression target y and features X
y = data_2d[:, voxel_idx]               # (nt,)
X = np.delete(data_2d, voxel_idx, axis=1)  # (nt, n_vox-1)

# 5. Standardize features
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# 6. Train linear SVR
svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
svr.fit(Xs, y)

# 7. Predict the time series of the selected voxel
y_pred = svr.predict(Xs)

# 8. Visualize: observed vs. predicted
plt.figure(figsize=(8,3))
plt.plot(y,       color='k', lw=1, label='Observed')
plt.plot(y_pred,  color='b', lw=1, label='Predicted')
plt.xlabel('Time (frames)')
plt.ylabel('Signal (a.u.)')
plt.title(f'Voxel {voxel_idx}: observed vs predicted\nr = {np.corrcoef(y, y_pred)[0,1]:.3f}')
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()

# 9. Save to your output directory
output_dir = os.path.join(
    OUTPUT_ROOT,
    format_output_name(f"voxel_pred_{s_id}_year{year}")
)
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, f"voxel_{voxel_idx}_timeseries.png"), dpi=300)
plt.close()
print(f"Saved comparison plot to {output_dir}")
