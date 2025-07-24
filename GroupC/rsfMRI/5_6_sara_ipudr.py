# File: 5_6_sara_ipudr.py

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from Util.config import OUTPUT_ROOT, dmn_names_33

def main():
    # 1. load the precomputed dataset
    data = np.load(os.path.join(OUTPUT_ROOT, 'fc_datasets', 'fc_data_16x3.npz'))
    fc      = data['fc']       # shape = (48, 528)
    subject = data['subject']  # shape = (48,)
    sara    = data['sara']     # shape = (48,)

    # 2. compute each subject's average SARA over 3 years
    subs = np.unique(subject)
    avg_sara = {s: sara[subject == s].mean() for s in subs}

    # 3. sort subjects by avg SARA ascending
    subs_sorted = sorted(subs, key=lambda s: avg_sara[s])
    n_subj = len(subs_sorted)  # should be 16

    # 4. build x-axis labels
    x = np.arange(n_subj)
    x_labels = [f"P{int(s):02d}" for s in subs_sorted]

    # 5. prepare ROI‑pair names
    pair_names = []
    n_rois = len(dmn_names_33)
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            pair_names.append(f"{dmn_names_33[i]}–{dmn_names_33[j]}")
    D = len(pair_names)  # 528

    # 6. prepare output directory with timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"{ts}_ipudr_fc_sara")
    os.makedirs(out_dir, exist_ok=True)

    # 7. for each ROI-pair, compute per-subject stats and plot
    for idx, name in enumerate(pair_names):
        means = []
        mins  = []
        maxs  = []
        for s in subs_sorted:
            vals = fc[subject == s, idx]  # three values
            means.append(vals.mean())
            mins.append(vals.min())
            maxs.append(vals.max())

        means = np.array(means)
        yerr_lower = means - np.array(mins)
        yerr_upper = np.array(maxs) - means
        yerr = np.vstack([yerr_lower, yerr_upper])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            x, means, yerr=yerr,
            fmt='-o', capsize=3, markersize=4, linewidth=1
        )
        ax.set_title(f"FC: {name}")
        ax.set_xlabel("Subject (sorted by avg SARA)")
        ax.set_ylabel("Correlation")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=90, fontsize='small')
        plt.tight_layout()

        # sanitize filename
        safe_name = name.replace(' ', '_').replace('/', '_').replace('–', '_')
        fname = f"pair_{idx+1:03d}_{safe_name}_ipudr.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)

    print(f"All {D} ROI-pair errorbar plots saved in:\n  {out_dir}")

if __name__ == "__main__":
    main()
