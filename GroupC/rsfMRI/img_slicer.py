import os
import nibabel as nib
from multiprocessing import Process
from Util.util_img import save_slices_along_axis
from Util.util_io import mri_path_niar
from Util.config import OUTPUT_ROOT, NIAR
import matplotlib
matplotlib.use('MacOSX')  # option: TkAgg (Tkinter GUI), MacOSX


# prepare target calls
def run(axis_name):
    s_id = "C01"
    r_id = "3"
    img = nib.load(mri_path_niar(NIAR, "C01", "3"))
    output = os.path.join(OUTPUT_ROOT, f'{axis_name}-axis')
    save_slices_along_axis(img, axis=axis_name, output_dir=output)


if __name__ == '__main__':
    axes = ['x', 'y', 'z']
    processes = [Process(target=run, args=(ax,)) for ax in axes]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
