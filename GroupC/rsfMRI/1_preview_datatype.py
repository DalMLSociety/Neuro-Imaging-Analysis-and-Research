import nibabel as nib
from Util.util_io import mri_path_niar
from Util.config import NIAR

s_id = "p01"
r_id = "1"
img = nib.load(mri_path_niar(NIAR, s_id, r_id))

# Basic shape and affine matrix
print("Image shape:", img.shape)
print("Affine matrix:\n", img.affine)

# Header information
hdr = img.header
print("Data type:", hdr.get_data_dtype())

zooms = hdr.get_zooms()
print("Voxel dimensions (mm):", zooms)

# Time resolution (TR)
tr = zooms[3] if len(zooms) > 3 else None
if tr is not None:
    print(f"Time resolution (TR): {tr} seconds")
else:
    print("Time resolution (TR): N/A")

# Total time frames
num_frames = img.shape[3] if len(img.shape) == 4 else None
print("Total time frames:", num_frames if num_frames else "N/A")

# Total scan duration
if tr and num_frames:
    total_seconds = tr * num_frames
    total_minutes = total_seconds / 60
    print(f"Total scan duration: {total_seconds:.1f} seconds ({total_minutes:.1f} minutes)")
