import os
import nibabel as nib
from nilearn import plotting
from util import get_mri_file_path
from config import OUTPUT_ROOT
import matplotlib
matplotlib.use('MacOSX')  # option: TkAgg (Tkinter GUI), MacOSX

# import dataset
file_path = get_mri_file_path(pre_path=['..', '..', '..'],
                              dataset_name='MRIData',
                              path=['sub-kaneff01', 'anat', 'sub-kaneff01_T1w.nii.gz'])
img = nib.load(file_path)

step = 5  # save the figure every 5 layer

# Iterate through the volume along the x-axis and save one sagittal slice every `step` layers
x_dir = os.path.join(OUTPUT_ROOT, 'x-axis')
os.makedirs(x_dir, exist_ok=True)
for x in range(0, img.shape[0], step):
    fig = plotting.plot_anat(img,
                             display_mode='x',
                             cut_coords=[x],
                             draw_cross=False)
    fig.savefig(os.path.join(x_dir, f'x_slice_{x:03d}.png'))
    fig.close()

# Iterate through the volume along the y-axis and save one coronal slice every `step` layers
y_dir = os.path.join(OUTPUT_ROOT, 'y-axis')
os.makedirs(y_dir, exist_ok=True)
for y in range(0, img.shape[1], step):
    fig = plotting.plot_anat(img,
                             display_mode='y',
                             cut_coords=[y],
                             draw_cross=False)
    fig.savefig(os.path.join(y_dir, f'y_slice_{y:03d}.png'))
    fig.close()

# Iterate through the volume along the z-axis and save one axial slice every `step` layers
z_dir = os.path.join(OUTPUT_ROOT, 'z-axis')
os.makedirs(z_dir, exist_ok=True)
for z in range(0, img.shape[2], step):
    fig = plotting.plot_anat(img,
                             display_mode='z',
                             cut_coords=[z],
                             draw_cross=False)
    fig.savefig(os.path.join(z_dir, f'z_slice_{z:03d}.png'))
    fig.close()
