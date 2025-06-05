import os
from typing import Sequence


# root_name: The root name of the repository (dataset)
# path: The specific
#
def get_mri_file_path(dataset_name, path: Sequence[str]):
    pre_path = ['..', '..', '..', '..']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *pre_path, dataset_name, *path)
