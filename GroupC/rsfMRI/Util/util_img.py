import os
from nibabel.affines import apply_affine
from nilearn import plotting


# Save anatomical slices (sagittal, coronal, axial)
# Arguments:
#   img: nibabel image object
#   axis: 'x' (sagittal), 'y' (coronal), or 'z' (axial)
#   output_dir: directory to save the output images
#   step: interval in voxel index to take slices
#   draw_cross: whether to draw cross-hairs on the image
def save_slices_along_axis(nib_img, axis, output_dir, step=5, draw_cross=False):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")

    dim = axis_map[axis]
    os.makedirs(output_dir, exist_ok=True)
    shape = nib_img.shape

    for idx in range(0, shape[dim], step):
        voxel = [0, 0, 0]
        voxel[dim] = idx
        mm_coord = apply_affine(nib_img.affine, voxel)[dim]

        fig = plotting.plot_anat(
            nib_img,
            display_mode=axis,
            cut_coords=[mm_coord],
            draw_cross=draw_cross
        )
        fName = os.path.join(output_dir, f"slice_{axis}_{idx:03d}.png")
        fig.savefig(fName, dpi=300)
        fig.close()

        print(f"[Saved] {fName}")
