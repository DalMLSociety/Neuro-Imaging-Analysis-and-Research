import os
import numpy as np
import pandas as pd

def is_aal3_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, nrows=1)
        cols = list(df.columns)
        return (cols and all(c.startswith("AAL3_") for c in cols))
    except Exception as e:
        print(f"[⚠️] Error reading {csv_path}: {e}")
        return False

def compute_fc_matrix(csv_path, common_cols=None):
    df = pd.read_csv(csv_path)
    if common_cols is not None:
        df = df[common_cols]
    corr = df.corr().values
    return corr

def flatten_fc_matrix(corr):
    fc_flat = corr[np.triu_indices_from(corr, k=1)]
    return fc_flat

def build_X_and_labels(control_folder, patient_folder, out_X, out_y):
    files_control = sorted([f for f in os.listdir(control_folder) if f.endswith('.csv') and is_aal3_csv(os.path.join(control_folder, f))])
    files_patient = sorted([f for f in os.listdir(patient_folder) if f.endswith('.csv') and is_aal3_csv(os.path.join(patient_folder, f))])

    if not files_control or not files_patient:
        print("[❌] هیچ فایل متناسب با AAL3 پیدا نشد! لطفاً فایل‌های خود را چک کنید.")
        raise RuntimeError("No suitable AAL3 CSV files found in one or both folders.")

    cols_control = pd.read_csv(os.path.join(control_folder, files_control[0])).columns
    cols_patient = pd.read_csv(os.path.join(patient_folder, files_patient[0])).columns
    common_columns = sorted(list(set(cols_control).intersection(set(cols_patient))))
    print(f"✅ Common columns ({len(common_columns)}): {common_columns[:5]} ...")

    X = []
    y = []
    for f in files_control:
        corr = compute_fc_matrix(os.path.join(control_folder, f), common_columns)
        fc_flat = flatten_fc_matrix(corr)
        X.append(fc_flat)
        y.append(0)
    for f in files_patient:
        corr = compute_fc_matrix(os.path.join(patient_folder, f), common_columns)
        fc_flat = flatten_fc_matrix(corr)
        X.append(fc_flat)
        y.append(1)

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] X.shape={X.shape}, y.shape={y.shape}")

    np.save(out_X, X)
    np.save(out_y, y)
    print(f"✅ Saved X ({X.shape}) to {out_X}✅ Saved y ({y.shape}) to {out_y}")

    return common_columns

def reshape_all_to_square(input_npy_path, output_npy_path):
    X = np.load(input_npy_path)
    if X.ndim != 2:
        print(f"❌ Wrong array shape in {input_npy_path}. Expected 2D but got {X.shape}")
        return
    N = int((1 + np.sqrt(1 + 8*X.shape[1])) / 2)
    X_square = np.zeros((X.shape[0], N, N))
    for i, fc_flat in enumerate(X):
        square = np.zeros((N, N))
        square[np.triu_indices(N, k=1)] = fc_flat
        square += square.T
        X_square[i] = square
    np.save(output_npy_path, X_square)
    print(f"✅ All samples saved in square format to {output_npy_path}")

# ------ فقط روی داده خام کار می‌کنیم (resid/residual نداریم)
CONTROL_DIR = "roi_time_series/aal3/control"
PATIENT_DIR = "roi_time_series/aal3/patient"
OUT_DIR = "roi_time_series"

os.makedirs(OUT_DIR, exist_ok=True)

common_columns = build_X_and_labels(
    CONTROL_DIR,
    PATIENT_DIR,
    os.path.join(OUT_DIR, "raw_fc.npy"),
    os.path.join(OUT_DIR, "raw_fc_labels.npy")
)

reshape_all_to_square(
    os.path.join(OUT_DIR, "raw_fc.npy"),
    os.path.join(OUT_DIR, "raw_fc_square.npy")
)

