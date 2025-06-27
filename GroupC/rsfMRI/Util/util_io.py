import os
from typing import Sequence
from datetime import datetime


# root_name: The root name of the repository (dataset)
# path: The specific
#
def mri_path(name: str, path: Sequence[str]):
    pre_path = ['..', '..', '..', '..']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *pre_path, name, *path)


def mri_path_niar(name: str, s_id: str, year: str):
    return mri_path(name, [mri_name(s_id, year)])


# s_id: C01-C16 or p01-p16
# r_id: resting-state run index (1, 2, 3)
def mri_name(s_id: str, r_id: str):
    return f"Denoised_{s_id}_rs-{r_id}_MNI.nii.gz"


def format_output_name(name: str):
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
