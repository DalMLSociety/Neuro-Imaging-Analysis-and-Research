## ICA scripts overview

This document summarizes the six ICA-focused Python scripts currently kept in the project root. Each entry explains the purpose, typical inputs/outputs, and when to use it.

### `sca_brain_pattern_ica_analysis.py`
- **Purpose**: Full ICA pipeline to identify SCA-related brain network biomarkers from rs-fMRI.
- **Key steps**: Load NIfTI data, prepare `NiftiMasker`, extract signals, run FastICA, compute statistics, generate spatial maps and summary visuals.
- **Input**: NIfTI files (default directory path is set inside the script/class).
- **Output**: Results in `sca_ica_biomarker_results/` (metrics, figures, component summaries).
- **Use when**: You want a comprehensive ICA biomarker discovery workflow end‑to‑end.

### `longitudinal_ica_compensation_analysis.py`
- **Purpose**: Map how individual SCA patients compensate over time (rs-1, rs-2, rs-3) and identify early biomarkers.
- **Key steps**: Organize longitudinal data, run PCA/ICA, derive compensation maps, evaluate early rs‑1 predictors of later outcomes, produce detailed reports/plots.
- **Input**: NIfTI files in a longitudinal structure (default `extracted data/raw_data`).
- **Output**: `longitudinal_ica_compensation_results/` (per‑subject maps, summaries, figures, JSON/CSV outputs).
- **Use when**: You need longitudinal compensation analysis and early biomarker discovery.

### `true_longitudinal_compensation_analysis.py`
- **Purpose**: Strict within‑subject longitudinal ICA analysis tracking each patient across all three timepoints.
- **Key steps**: Identify subjects with complete data, create consistent masker, run ICA, model per‑subject trajectories (slopes, R²), summarize group patterns.
- **Input**: NIfTI files (default `extracted data/raw_data`).
- **Output**: `true_longitudinal_results/` (trajectory metrics, component trends, summary visuals).
- **Use when**: You want per‑subject compensation trajectories with strict longitudinal inclusion.

### `optimized_sca_multithread_analysis.py`
- **Purpose**: High‑performance ICA pipeline using multi‑processing and memory‑aware settings.
- **Key steps**: Chunked file loading, optimized `NiftiMasker`, PCA→ICA, classification metrics, parallel processing tuned to system resources.
- **Input**: NIfTI files (default `extracted data/raw_data`).
- **Output**: `optimized_sca_results/` (optimized run artifacts, metrics, figures, JSON/CSV).
- **Use when**: You need faster runs on larger datasets or constrained RAM/CPU.

### `simple_sca_pattern_analysis.py`
- **Purpose**: Lightweight/quick ICA analysis for fast iteration and sanity checks.
- **Key steps**: Load a subset of files, simple masker, PCA→ICA, identify significant components, optional quick classification and basic plots.
- **Input**: NIfTI files (you can limit `max_files`).
- **Output**: Console summaries and saved plots/CSVs (paths defined in script).
- **Use when**: You want a quick, lower‑resource pass to validate signals and pipeline settings.

### `interpret_ica_components.py`
- **Purpose**: Interpret ICA components by mapping them to anatomical and functional networks.
- **Key steps**: Load ICA results, fetch atlases (Schaefer, Harvard‑Oxford), score component‑to‑region/network associations, summarize top biomarkers.
- **Input**: Outputs from an ICA run (e.g., `ica_components.csv`, analysis JSON) in the specified `results_dir`.
- **Output**: Annotated component summaries, network/region attributions, optional figures.
- **Use when**: You want anatomical/functional meaning for discovered ICA components.

## Notes
- If your data/results folders are in `backup/`, update each script’s `data_dir`/`results_dir` accordingly or move the needed folders back alongside the scripts.
- These scripts rely on `nilearn`, `scikit‑learn`, `numpy`, `pandas`, and related scientific libraries. See `requirements_ica.txt` or `requirements_optimized.txt`.

