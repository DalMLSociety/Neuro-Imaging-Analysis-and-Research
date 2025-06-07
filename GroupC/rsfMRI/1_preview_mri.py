from nilearn import plotting
import nibabel as nib
from Util.util_io import mri_path_niar
from Util.config import NIAR


s_id = "C01"
r_id = "3"
img = nib.load(mri_path_niar(NIAR, s_id, r_id))

# Extract the first time frame (3D volume) for visualization
img_3d = nib.Nifti1Image(img.get_fdata()[..., 0], img.affine)

# Launch interactive HTML viewer in your web browser
viewer = plotting.view_img(img_3d, title=f"{s_id} rs-{r_id}")
viewer.open_in_browser()
