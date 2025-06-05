import os
import numpy as np
from nilearn import plotting, datasets
from Util.config import OUTPUT_ROOT
import matplotlib

matplotlib.use('Agg')


def extract_strong_connections(matrix, labels, threshold=0.6):
    strong_pairs = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i, j]
            if abs(value) >= threshold:
                strong_pairs.append((labels[i], labels[j], value))
    return sorted(strong_pairs, key=lambda x: -abs(x[2]))


# load matrix and label
matrix_path = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001.npy')
matrix = np.load(matrix_path)
atlas = datasets.fetch_atlas_aal()
labels = atlas.labels

strong = extract_strong_connections(matrix, labels, threshold=0.7)

# save to text
output_txt = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001_strong_pairs.txt')
with open(output_txt, 'w') as f:
    for a, b, v in strong:
        f.write(f"{a} ↔ {b}: {v:.2f}\n")

# visualize
fig = plotting.plot_matrix(matrix,
                           figure=(10, 8),
                           vmin=-1.0, vmax=1.0,
                           colorbar=True,
                           title='Strong Functional Connectivity')
fig_path = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001_strong.png')
fig.figure.savefig(fig_path, dpi=300)

print(f"Strong connectivity pairs saved to: {output_txt}")
print(f"Figure saved to: {fig_path}")
