import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, input_data
from scipy.signal import coherence
from Util.util_io import get_mri_file_path
from Util.config import OUTPUT_ROOT
from PIL import Image
from io import BytesIO

label_a = "Hippocampus_L"
label_b = "Angular_L"
output_dir = os.path.join(OUTPUT_ROOT, "fc2")
save_prefix = "connectivity_pair"
anat_path = get_mri_file_path(dataset_name="MRIData",
                              path=["sub-kaneff01", "anat", "sub-kaneff01_T1w.nii.gz"])
bold_path = get_mri_file_path(dataset_name="MRIData",
                              path=["sub-kaneff01", "func", "sub-kaneff01_task-effloc_run-001_bold.nii.gz"])

bold_img = nib.load(bold_path)
anat_img = nib.load(anat_path)
atlas = datasets.fetch_atlas_aal()
masker = input_data.NiftiLabelsMasker(labels_img=atlas.maps, standardize=True, detrend=True)
region_ts = masker.fit_transform(bold_img)
labels = atlas.labels

# extract time series from the two ROIs
idx_a = np.where(np.array(labels) == label_a)[0][0]
idx_b = np.where(np.array(labels) == label_b)[0][0]
ts_a = region_ts[:, idx_a]
ts_b = region_ts[:, idx_b]

# compute functional connectivity metrics
pearson_corr = np.corrcoef(ts_a, ts_b)[0, 1]
f_coh, coh = coherence(ts_a, ts_b)

# create 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Show both ROIs on anatomical background
# Use plot_roi to draw ROI overlays and render into memory
roi_fig = plotting.plot_roi(atlas.maps, bg_img=anat_img,
                            display_mode='ortho', cut_coords=None)

# save to memory buffer
buf = BytesIO()
roi_fig.frame_axes.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
roi_fig.close()
buf.seek(0)

# convert to image and draw on subplot
img_roi = Image.open(buf)
axes[0, 0].imshow(img_roi)
axes[0, 0].axis('off')
axes[0, 0].set_title(f"(a) ROI A: {label_a} | ROI B: {label_b}")

# (b) Plot time series of both ROIs
axes[0, 1].plot(ts_a, label=f"A: {label_a}", color='blue')
axes[0, 1].plot(ts_b, label=f"B: {label_b}", color='green')
axes[0, 1].set_title("(b) ROI Time Series")
axes[0, 1].set_xlabel("Time")
axes[0, 1].set_ylabel("Signal")
axes[0, 1].legend()

# (c) Scatter plot of ROI A vs ROI B
axes[1, 0].scatter(ts_a, ts_b, s=10, alpha=0.6)
axes[1, 0].set_xlabel(f"A: {label_a}", color='blue')
axes[1, 0].set_ylabel(f"B: {label_b}", color='green')
axes[1, 0].set_title("(c) Scatter Plot")

# (d) Show connectivity metrics as text
axes[1, 1].axis("off")
text = f"(d) Functional Connectivity\n\n" \
       f"• Time-domain similarity = {pearson_corr:.4f} (Pearson)\n" \
       f"• Frequency-domain similarity = {np.max(coh):.4f} (Coherence)"
axes[1, 1].text(0, 0.5, text, fontsize=12, va="center", ha="left")

# save the final figure
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, f"{save_prefix}_{label_a}_{label_b}.png")
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved to: {out_path}")
