import os
import matplotlib.pyplot as plt
from nilearn import input_data
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT

# ----- Input -----
subject_ids = ["C01", "C05", "C08", "C15", "p02", "p06", "p11", "p16"]
years = ["1", "2", "3"]
colors = ["#d62728", "#2ca02c", "#1f77b4"]

regions = [
    {"label": "PCC",         "coord": (0, -52, 24)},
    {"label": "MPFC",        "coord": (0, 52, -6)},
    {"label": "Amygdala",    "coord": (-20, -4, -16)},
    {"label": "Hippocampus", "coord": (-24, -18, -18)},
    {"label": "V1",          "coord": (0, -90, 0)},
    {"label": "DLPFC",       "coord": (-40, 30, 30)},
    {"label": "Insula",      "coord": (38, 20, 4)},
    {"label": "Thalamus",    "coord": (0, -15, 10)},
]

# ----- Helper Function: extract all time series for a subject -----
def extract_subject_time_series(s_id):
    subject_ts = []
    for region in regions:
        coord = region["coord"]
        region_ts = []
        for year in years:
            img = nib.load(mri_path_niar(NIAR, s_id, year))
            masker = input_data.NiftiSpheresMasker([coord], radius=4, detrend=True, standardize=True)
            ts = masker.fit_transform(img)
            region_ts.append(ts.ravel())
        subject_ts.append(region_ts)  # 3 years x T
    return subject_ts  # 8 regions x 3 years x T

# ----- Run in parallel -----
print("Extracting time series in parallel...")
all_data = Parallel(n_jobs=-1)(delayed(extract_subject_time_series)(s_id) for s_id in subject_ids)

# ----- Plotting -----
output_path = os.path.join(OUTPUT_ROOT, format_output_name("bold-timeseries_grid_overlay.png"))
fig, axes = plt.subplots(nrows=len(subject_ids), ncols=len(regions), figsize=(24, 2.6 * len(subject_ids)), sharex=True, sharey=True)
fig.suptitle("BOLD Time Series per Region (Overlaid) — Multiple Subjects", fontsize=18)

for row, s_id in enumerate(subject_ids):
    subject_ts = all_data[row]
    for col, region in enumerate(regions):
        ax = axes[row, col] if len(subject_ids) > 1 else axes[col]
        for i, ts in enumerate(subject_ts[col]):
            ax.plot(ts, color=colors[i], linewidth=1.2, label=f"Year {years[i]}" if row == 0 and col == 0 else None)
        if row == 0:
            ax.set_title(region["label"], fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{s_id}", fontsize=11)
        T = len(subject_ts[col][0])
        ax.set_xticks([0, T // 2, T - 1])
        ax.set_xticklabels(["0", f"{T // 2}", f"{T - 1}"], fontsize=7)
        ax.set_xlabel("TR", fontsize=8)

# ----- Legend -----
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10, frameon=False)

# ----- Final touches -----
plt.tight_layout()
plt.subplots_adjust(top=0.94, bottom=0.08, hspace=0.25, wspace=0.25)
plt.savefig(output_path, dpi=300)
print(f"[Saved] {output_path}")
