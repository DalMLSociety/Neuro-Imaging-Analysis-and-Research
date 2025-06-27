from nilearn import plotting
import nibabel as nib
from Util.util_io import mri_path_niar
from Util.config import NIAR


s_id = "p03"
year = "1"
img = nib.load(mri_path_niar(NIAR, s_id, year))

# Extract the first time frame (3D volume) for visualization
frame_t = 0
img_3d = nib.Nifti1Image(img.get_fdata()[..., frame_t], img.affine)

# Launch interactive HTML viewer in your web browser
viewer = plotting.view_img(img_3d, title=f"{s_id} rs-{year}")
viewer.open_in_browser()
