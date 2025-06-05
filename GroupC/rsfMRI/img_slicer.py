import os
import nibabel as nib
from multiprocessing import Process
from Util.util_img import save_slices_along_axis
from Util.util_io import get_mri_file_path
from Util.config import OUTPUT_ROOT
import matplotlib
matplotlib.use('MacOSX')  # option: TkAgg (Tkinter GUI), MacOSX


# prepare target calls
def run(axis_name):
    file_path = get_mri_file_path(dataset_name='MRIData',
                                  path=['sub-kaneff01', 'anat', 'sub-kaneff01_T1w.nii.gz'])
    img = nib.load(file_path)
    output = os.path.join(OUTPUT_ROOT, f'{axis_name}-axis')
    save_slices_along_axis(img, axis=axis_name, output_dir=output)


if __name__ == '__main__':
    axes = ['x', 'y', 'z']
    processes = [Process(target=run, args=(ax,)) for ax in axes]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
