import os
import numpy as np
from nilearn import plotting
from Util.config import OUTPUT_ROOT
import matplotlib
matplotlib.use('Agg')

# Load the matrix
matrix_path = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001.npy')
save_path = os.path.join(OUTPUT_ROOT, 'fc', 'connectivity_matrix_run001.png')
matrix = np.load(matrix_path)

# Generate and save the connectivity matrix figure
display = plotting.plot_matrix(matrix,
                               figure=(10, 8),
                               vmin=-1.0, vmax=1.0,
                               colorbar=True,
                               title='Functional Connectivity')
display.figure.savefig(save_path, dpi=300)

print(f"Connectivity matrix figure saved to: {matrix_path}")
