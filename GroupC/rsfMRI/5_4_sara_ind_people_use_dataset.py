# File: 5_4_sara_ind_people_use_dataset.py

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from Util.config import OUTPUT_ROOT, dmn_names_33

def main():
    # 1. load the precomputed dataset
    data = np.load(os.path.join(OUTPUT_ROOT, 'fc_datasets', 'fc_data_16x3.npz'))
    fc    = data['fc']    # shape = (48, 528)
    sara  = data['sara']  # shape = (48,)

    # 2. sort all 48 samples by sara ascending
    order = np.argsort(sara)
    fc_sorted   = fc[order, :]
    sara_sorted = sara[order]

    # 3. build ROI-pair names list
    #    dmn_names_33 is a list of 33 ROI names
    pair_names = []
    n_rois = len(dmn_names_33)
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            pair_names.append(f"{dmn_names_33[i]}–{dmn_names_33[j]}")
    D = len(pair_names)  # should be 528

    # 4. prepare output folder with timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"{ts}_fc_sara")
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(len(sara_sorted))

    # 5. plot one figure per ROI-pair
    for idx, name in enumerate(pair_names):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, fc_sorted[:, idx], marker='o', linestyle='-')
        ax.set_title(f"FC: {name}")
        ax.set_xlabel("Samples (sorted by SARA)")
        ax.set_ylabel("Correlation")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i+1}" for i in x], rotation=90, fontsize='small')
        plt.tight_layout()

        # sanitize filename
        fname = f"pair_{idx+1:03d}_{name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)

    print(f"All {D} ROI-pair plots saved in:\n  {out_dir}")

if __name__ == "__main__":
    main()
