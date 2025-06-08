import nibabel as nib
from Util.util_io import mri_path_niar
from Util.config import NIAR

# Image shape: (91, 109, 91, 236)
# Affine matrix:
#  [[  -2.    0.    0.   90.]
#  [   0.    2.    0. -126.]
#  [   0.    0.    2.  -72.]
#  [   0.    0.    0.    1.]]
# X range (mm): [np.float64(-90.0), np.float64(90.0)]
# Y range (mm): [np.float64(-126.0), np.float64(90.0)]
# Z range (mm): [np.float64(-72.0), np.float64(108.0)]
# Data type: float32
# Voxel dimensions (mm): (np.float32(2.0), np.float32(2.0), np.float32(2.0), np.float32(2.0))
# Time resolution (TR): 2.0 seconds
# Total time frames: 236
# Total scan duration: 472.0 seconds (7.9 minutes)


# For each axis (x, y, z), get voxel index range and map to world coords
def get_axis_world_range(affine, shape, axis):
    idx_min = [0, 0, 0]
    idx_max = [s - 1 for s in shape]

    idx_min[axis] = 0
    idx_max[axis] = shape[axis] - 1

    coord_min = nib.affines.apply_affine(affine, idx_min)[axis]
    coord_max = nib.affines.apply_affine(affine, idx_max)[axis]

    return sorted([coord_min, coord_max])  # ensure min < max


s_id = "p01"
r_id = "1"
img = nib.load(mri_path_niar(NIAR, s_id, r_id))

# Basic shape and affine matrix
affine = img.affine
shape = img.shape
print("Image shape:", shape)
print("Affine matrix:\n", affine)
# 3D shape
x_range = get_axis_world_range(affine, shape[:3], axis=0)
y_range = get_axis_world_range(affine, shape[:3], axis=1)
z_range = get_axis_world_range(affine, shape[:3], axis=2)
print(f"X range (mm): {x_range}")
print(f"Y range (mm): {y_range}")
print(f"Z range (mm): {z_range}")

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
