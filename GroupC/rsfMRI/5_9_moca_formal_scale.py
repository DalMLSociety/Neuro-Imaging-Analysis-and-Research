# File: 5_9_moca_formal_scale.py

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from Util.config import OUTPUT_ROOT, dmn_names_33

def main():
    # 1. load FC dataset
    data = np.load(os.path.join(OUTPUT_ROOT, 'fc_datasets', 'fc_data_16x3.npz'))
    fc      = data['fc']       # shape = (48, 528)
    subject = data['subject']  # shape = (48,)
    sara    = data['sara']     # shape = (48,)

    # 2. load MOCA from clinical Excel and compute per-sample MOCA
    import pandas as pd
    df = pd.read_excel('demoSCA7.xlsx')
    df['subject'] = df.index // 3 + 1
    df['year']    = df.groupby('subject').cumcount() + 1

    # compute average MOCA per subject
    subs = np.unique(df['subject'])
    avg_moca = np.array([df.loc[df['subject']==s, 'moca'].mean() for s in subs])

    # 3. sort subjects by avg MOCA
    order_subs     = np.argsort(avg_moca)
    subs_sorted    = subs[order_subs]
    avg_moca_sorted= avg_moca[order_subs]
    n_subj         = len(subs_sorted)

    # 4. build fc_means, mins, maxs per subject
    D = fc.shape[1]
    fc_means = np.zeros((n_subj, D))
    fc_mins  = np.zeros((n_subj, D))
    fc_maxs  = np.zeros((n_subj, D))
    for i, s in enumerate(subs_sorted):
        vals = fc[subject == s, :]  # shape (3, D)
        fc_means[i] = vals.mean(axis=0)
        fc_mins[i]  = vals.min(axis=0)
        fc_maxs[i]  = vals.max(axis=0)

    # 5. compute regression slopes (vs avg_moca)
    slopes = np.array([
        np.polyfit(avg_moca_sorted, fc_means[:, j], 1)[0]
        for j in range(D)
    ])

    # 6. rank all ROI-pairs by |slope|
    sorted_idx = np.argsort(np.abs(slopes))[::-1]
    rank_map   = {idx: rank+1 for rank, idx in enumerate(sorted_idx)}

    # 7. ROI-pair names
    pair_names = []
    n_rois = len(dmn_names_33)
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            pair_names.append(f"{dmn_names_33[i]}–{dmn_names_33[j]}")

    # 8. prepare output dir
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"{ts}_moca_formal_scale")
    os.makedirs(out_dir, exist_ok=True)

    # Use avg MOCA as x-axis
    x = avg_moca_sorted

    # 9. plot all ROI-pairs
    for idx, name in enumerate(pair_names):
        means = fc_means[:, idx]
        mins  = fc_mins[:, idx]
        maxs  = fc_maxs[:, idx]
        yerr  = np.vstack([means - mins, maxs - means])

        # regression line
        m, b = np.polyfit(x, means, 1)
        y_fit = m * x + b

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            x, means, yerr=yerr,
            fmt='o-', capsize=3, markersize=4, linewidth=1,
            label='mean ± range'
        )
        ax.plot(x, y_fit, 'r--', linewidth=1.5, label=f'slope={m:.3f}')

        # title and labels
        rank = rank_map[idx]
        ax.set_title(f"#{rank} {name}")
        ax.set_xlabel("Average MOCA")
        ax.set_ylabel("Correlation")

        # x‑ticks at each subject’s avg MOCA
        ax.set_xticks(x)
        ax.set_xticklabels([f"{val:.1f}" for val in x], rotation=90, fontsize='small')

        ax.legend(fontsize='small')
        plt.tight_layout()

        # filename with rank prefix
        safe = name.replace(' ', '_').replace('/', '_').replace('–', '_')
        fname = f"#{rank:03d}_pair_{idx+1:03d}_{safe}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)

    print(f"All {D} ROI-pair plots saved in:\n  {out_dir}")
    print("X‑axis positions now correspond to each subject’s true average MOCA.")

if __name__ == "__main__":
    main()
