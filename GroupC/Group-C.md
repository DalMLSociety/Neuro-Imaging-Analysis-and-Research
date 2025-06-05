## Group-C

### File Structure Convention

Before running any Python code in this repository, please make sure to follow the directory structure below:

```
MRI-Research/
├── <Neuro-Imaging-Analysis-and-Research>
├── <dataset-folder-1>/
├── <dataset-folder-2>/
└── ...
```

- The entire repository must be placed under a folder named **`MRI-Research/`**.
- Any dataset folders should be placed **directly inside** the `MRI-Research` directory (i.e., at the same level as the repository files).
- This structure makes it easier to set up relative paths for importing datasets, especially since datasets cannot be pushed to the repository.

> **Note:** Datasets are not included in this repository. Please download them manually and place them under the `MRI-Research/` folder.


### Dependency

For dependency of the project, run:
```bash
  pip install -r dependency.txt
```