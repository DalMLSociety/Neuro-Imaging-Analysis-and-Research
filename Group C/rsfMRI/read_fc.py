import numpy as np
from nilearn import plotting
import matplotlib
matplotlib.use('Agg')

# Load the matrix
matrix = np.load('connectivity_matrix_run001.npy')

# Generate and save the connectivity matrix figure
display = plotting.plot_matrix(matrix,
                               figure=(10, 8),
                               vmin=-1.0, vmax=1.0,
                               colorbar=True,
                               title='Functional Connectivity')

# Save the figure instead of calling plotting.show()
display.figure.savefig('fc_matrix.png', dpi=300)
display.close()
