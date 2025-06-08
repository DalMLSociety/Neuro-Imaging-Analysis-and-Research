import os
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.decomposition import CanICA
from nilearn.masking import compute_epi_mask
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

from Util.util_io import mri_path_niar, format_output_name
from Util.config import NIAR, OUTPUT_ROOT

# Load your fMRI image
s_id = "C01"
r_id = "3"
img = nib.load(mri_path_niar(NIAR, s_id, r_id))

# Compute brain mask
mask_img = compute_epi_mask(img)

# Initialize and fit CanICA
canica = CanICA(
    n_components=20,
    mask=mask_img,
    smoothing_fwhm=6,
    standardize=True,
    detrend=True,
    random_state=0,
    memory="nilearn_cache",
    memory_level=2,
    verbose=10
)
canica.fit(img)

# Extract components
components_img = canica.components_img_

output_dir = os.path.join(OUTPUT_ROOT, format_output_name(f"ica_{s_id}_{r_id}"))
os.makedirs(output_dir, exist_ok=True)

# Save each ICA component as image
for i in range(components_img.shape[-1]):
    display = plot_stat_map(
        index_img(components_img, i),
        title=f"ICA Component #{i}",
        display_mode="z",
        cut_coords=5,
        colorbar=True
    )
    fname = os.path.join(output_dir, f"ica_component_{i:02d}.png")
    display.savefig(fname, dpi=300)
    display.close()

print(f"All ICA components saved to {output_dir}")
