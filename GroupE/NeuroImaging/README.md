# Neuro-Imaging Analysis and Research (NIAR) - Group E

A comprehensive Python pipeline for analyzing neuroimaging data, focusing on **functional connectivity (FC)**, **residualized FC**, and graph-based machine learning models for classification and longitudinal prediction in neurological disorders.

## Overview
This repository contains tools for:
- Preprocessing ROI time series and building functional connectivity matrices.
- Generating **raw** and **residualized** FC features.
- Extracting statistical and graph-based features from FC networks.
- Training and evaluating both **classical ML models** and **graph neural networks (DAST-GCN)**.
- Visualizing connectivity patterns and model results.

The workflow supports **AAL3 atlas data** and is adaptable to other atlases.

---

## Project Structure

├── analysis.py                   # Statistical analysis and utility functions  
├── build_fc_matrices.py          # Build raw FC matrices from ROI time series  
├── build_fc_matrices_residFc.py  # Build residualized FC matrices + label alignment  
├── config.py                     # App/config flags, optional libs, Streamlit setup  
├── features.py                   # Static/dynamic FC features, sliding-window dFC  
├── gnn_models.py                 # MLP/GCN + DAST forecaster architectures  
├── labels.py                     # Generate labels.npy from folder structure  
├── lme.py                        # Linear Mixed-Effects residualization pipeline  
├── main.py                       # Streamlit app (end-to-end exploration & ML)  
├── ml.py                         # Classical ML (SVM/RF) with GroupKFold CV  
├── train_dast.py                 # K-Fold baselines + DAST helpers/evaluation  
├── utils.py                      # IO helpers, flexibility metric, misc utilities  
├── visualization.py              # 3D graph (Plotly) + dFC animation (moviepy)  
├── roi_time_series/              # Input time series & generated FC (.npy/.csv)  
└── results/                      # Saved outputs, metrics, figures, artifacts  


---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DalMLSociety/Neuro-Imaging-Analysis-and-Research.git
   cd Neuro-Imaging-Analysis-and-Research

2. **Create a virtual environment**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows:  env\Scripts\activate

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt

##  Usage
1. Prepare ROI time series
Place your .csv time series files in:
roi_time_series/aal3/control/   # Control group
roi_time_series/aal3/patient/   # Patient group

2. Build functional connectivity matrices
- Raw FC
    ```bash
    python build_fc_matrices.py
- Residualized FC
    ```bash
    python build_fc_matrices_residFc.py
3. Run the full pipeline
    ```bash
    python main.py
This will:
- Load FC data

- Extract features

- Train classical ML models and/or GNN models

- Save metrics and visualizations to results/
##  Models
1. Classical ML: SVM, Random Forest, Logistic Regression

2. Graph Neural Networks: DAST-GCN

3. Statistical Models: Linear Mixed-Effects (LME)

## Visualization
visualization.py supports:

- FC heatmaps

- Graph visualizations

- Performance metric plots
## Notes
- The code assumes AAL3 atlas-based ROI labels by default.

- config.py contains paths and hyperparameters; adjust before running.

- Large .npy files (FC matrices) are not included in the repo; regenerate them via the preprocessing scripts.

## License
MIT License