import os
from typing import Sequence


# root_name: The root name of the repository (dataset)
# path: The specific
#
def get_mri_file_path(root_name, path: Sequence[str]):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, root_name, *path)
