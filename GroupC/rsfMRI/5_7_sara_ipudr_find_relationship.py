# File: 5_7_sara_ipudr_regression.py

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from Util.config import OUTPUT_ROOT, dmn_names_33

def main():
    # 1. load dataset
    data = np.load(os.path.join(OUTPUT_ROOT, 'fc_datasets', 'fc_data_16x3.npz'))
    fc      = data['fc']       # (48, 528)
    subject = data['subject']  # (48,)
    sara    = data['sara']     # (48,)

    # 2. compute average SARA per subject
    subs = np.unique(subject)
    avg_sara = np.array([sara[subject == s].mean() for s in subs])

    # 3. sort subjects by avg SARA
    order_subs = np.argsort(avg_sara)
    subs_sorted = subs[order_subs]
    avg_sara_sorted = avg_sara[order_subs]
    n_subj = len(subs_sorted)

    # 4. prepare fc_means matrix: subjects × ROI-pairs
    D = fc.shape[1]
    fc_means = np.zeros((n_subj, D))
    fc_mins  = np.zeros((n_subj, D))
    fc_maxs  = np.zeros((n_subj, D))
    for i, s in enumerate(subs_sorted):
        vals = fc[subject == s, :]  # shape (3, D)
        fc_means[i, :] = vals.mean(axis=0)
        fc_mins[i, :]  = vals.min(axis=0)
        fc_maxs[i, :]  = vals.max(axis=0)

    # 5. compute regression slopes for each ROI-pair
    slopes = np.zeros(D)
    for j in range(D):
        m, b = np.polyfit(avg_sara_sorted, fc_means[:, j], 1)
        slopes[j] = m

    # 6. identify top N absolute slopes and rank them
    top_n = 10  # change to desired number of top slopes
    sorted_idx_by_slope = np.argsort(np.abs(slopes))[::-1]
    top_idxs = sorted_idx_by_slope[:top_n]
    # map each top idx to its rank
    rank_map = {idx: rank+1 for rank, idx in enumerate(top_idxs)}

    # 7. build ROI-pair names
    pair_names = []
    n_rois = len(dmn_names_33)
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            pair_names.append(f"{dmn_names_33[i]}–{dmn_names_33[j]}")

    # 8. prepare output directory
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"{ts}_ipudr_fc_sara_reg")
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(n_subj)

    # 9. plot each ROI-pair with error bars and regression line
    for idx, name in enumerate(pair_names):
        means = fc_means[:, idx]
        mins  = fc_mins[:, idx]
        maxs  = fc_maxs[:, idx]
        yerr = np.vstack([means - mins, maxs - means])

        m = slopes[idx]
        b = np.polyfit(avg_sara_sorted, means, 1)[1]
        y_fit = m * avg_sara_sorted + b

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            x, means, yerr=yerr,
            fmt='-o', capsize=3, markersize=4, linewidth=1,
            label='mean ± range'
        )
        ax.plot(x, y_fit, 'r--', linewidth=1.5, label=f'regress (slope={m:.3f})')

        ax.set_title(f"FC: {name}")
        ax.set_xlabel("Subject (sorted by avg SARA)")
        ax.set_ylabel("Correlation")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{int(s):02d}" for s in subs_sorted], rotation=90, fontsize='small')
        ax.legend(fontsize='small')
        plt.tight_layout()

        # add ranking tag if in top slopes
        tag = ''
        if idx in rank_map:
            tag = f"_topslope!!!!!!!!!!!!!!!!!!!!!!!{rank_map[idx]}"
        safe = name.replace(' ', '_').replace('/', '_').replace('–', '_')
        fname = f"pair_{idx+1:03d}_{safe}{tag}.png"

        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)

    print(f"Plots with regression saved in:\n  {out_dir}")
    print(f"Top {top_n} ROI-pairs by slope are tagged with rank in filenames.")

if __name__ == "__main__":
    main()
