import math
import os
import numpy as np
import pandas as pd
import torch
import warnings; warnings.filterwarnings('ignore')
import sys, os
from sklearn.model_selection import StratifiedKFold

#sys.stderr = open(os.devnull, 'w')

import pandas as pd
import numpy as np
import os

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,accuracy_score, f1_score, roc_auc_score, recall_score, precision_score,
    r2_score, mean_squared_error
)
from sklearn.model_selection import train_test_split
from gnn_models import DASTForecaster, MLPBaseline, BrainGCN,MLPBaselineImproved

# ================================
# STEP 0: Create ts_raw.csv
# ================================
def create_combined_ts_csv(control_dir, patient_dir, output_path):
    all_dfs = []
    for dir_path in [control_dir, patient_dir]:
        if not os.path.exists(dir_path): continue
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(dir_path, file))
                all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, axis=1)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"✅ ts_raw.csv created at: {output_path}")

ts_raw_path = "results/ts_raw.csv"
if not os.path.exists(ts_raw_path):
    create_combined_ts_csv(
        "roi_time_series/aal3/control", "roi_time_series/aal3/patient", ts_raw_path
    )
else:
    print("📁 ts_raw.csv already exists.")

# ================================
# SAVE MODEL OUTPUT
# ================================
def save_output(output_tensor, out_path):
    np.save(out_path, output_tensor.detach().cpu().numpy())
    print(f"✅ Output saved to: {out_path}")

# ================================
# EVAL METRICS
# ================================
def eval_metrics(pred, true):

    pred = np.array(pred)
    true = np.array(true)

    if pred.ndim == 2 and pred.shape[1] == 2:
        pred = np.argmax(pred, axis=1)
    elif pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:,0]
    else:
        pred = pred.reshape(-1)
    true = true.reshape(-1)

    results = {}
    
    is_binary = set(np.unique(true)) <= {0,1}
    pred_bin = (pred >= 0.5).astype(int) if is_binary else None

    # ------ CLASSIFICATION METRICS ------
    if is_binary:
        results['accuracy'] = accuracy_score(true, pred_bin)
        results['f1'] = f1_score(true, pred_bin)
        results['recall'] = recall_score(true, pred_bin)
        results['precision'] = precision_score(true, pred_bin)
        try:
            results['auc'] = roc_auc_score(true, pred)
        except:
            results['auc'] = None
    else:
        results['accuracy'] = None
        results['f1'] = None
        results['recall'] = None
        results['precision'] = None
        results['auc'] = None

    # ------ REGRESSION METRICS ------
    results['r2'] = r2_score(true, pred)
    results['rmse'] = np.sqrt(mean_squared_error(true, pred))

    # PRINT
    print("======== Validation Metrics ========")
    if is_binary:
        print(f"Accuracy : {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall   : {results['recall']:.4f}")
        print(f"F1-score : {results['f1']:.4f}")
        print(f"AUC      : {results['auc'] if results['auc'] is not None else '-'}")
    print(f"R2-score : {results['r2']:.4f}")
    print(f"RMSE     : {results['rmse']:.4f}")
    print("====================================")
    return results



def eval_metrics_dast(pred, true, windows_per_subject=None, aggregate='mean'):
    """
    Evaluates DAST model output against true labels, ensuring sizes match by aggregation.

    Args:
        pred: Model predictions (array-like, shape [N_windows, ...])
        true: True labels (array-like, shape [N_subjects])
        windows_per_subject: تعداد windowها برای هر subject (int یا list[int])
        aggregate: روش تجمیع، 'mean' برای مقدار میانگین، 'vote' برای majority vote

    Returns:
        metrics dict
    """

    import numpy as np

    pred = np.array(pred)
    true = np.array(true).reshape(-1)


    if windows_per_subject is None:
        win_per_subj = pred.shape[0] // true.shape[0]
    else:
        win_per_subj = windows_per_subject


    pred_subject = []
    for i in range(len(true)):
        sub_pred = pred[i*win_per_subj:(i+1)*win_per_subj]
        if aggregate == 'mean':
            pred_val = np.mean(sub_pred)
        elif aggregate == 'vote':
            pred_bin = (sub_pred >= 0.5).astype(int)
            counts = np.bincount(pred_bin)
            pred_val = np.argmax(counts)
        else:
            raise ValueError('aggregate must be "mean" or "vote"')
        pred_subject.append(pred_val)
    pred_subject = np.array(pred_subject)


    metrics = eval_metrics(pred_subject, true)
    return metrics
# ================================
# Split data to train/val
# ================================
def split_data(X, y, val_size=0.2, random_state=42):
    # Check if binary for stratify
    stratify = y if set(np.unique(y)) <= {0,1} else None
    return train_test_split(X, y, test_size=val_size, stratify=stratify, random_state=random_state)



# ================================
# Run MLP or GCN
# ================================
import numpy as np
import torch

def run_baseline(X_path, y_path, label, out_file, model_type='mlp', val_split=True):
    X = np.load(X_path)
    y = np.load(y_path).reshape(-1)  # Ensure 1D

    if val_split:
        X_train, X_val, y_train, y_val = split_data(X, y)
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    if model_type == 'gcn':

        if X_val.ndim == 3:

            num_samples, N, N2 = X_val.shape
            if N != N2:
                raise ValueError(f"❌ Input matrix must be square (got {N}x{N2})")
            tensor_val = torch.tensor(X_val).float()
        elif X_val.ndim == 2:

            N = int(np.sqrt(X_val.shape[1]))
            if N * N != X_val.shape[1]:
                raise ValueError(f"❌ Invalid shape: {X_val.shape[1]} can't reshape to square.")
            tensor_val = torch.tensor(X_val).float().reshape(-1, N, N)
        else:
            raise ValueError(f"❌ Unsupported shape for GCN: {X_val.shape}")


        model = BrainGCN(in_feats=1, h_feats=16, n_classes=1)
        edge_index = torch.combinations(torch.arange(N), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  
        batch = torch.zeros(N, dtype=torch.long)
        node_features = tensor_val.mean(dim=2)  
        all_outputs = []
        model.eval()
        with torch.no_grad():
            for nf in node_features:
                out = model(nf.unsqueeze(1), edge_index, batch)
                all_outputs.append(out.squeeze().cpu().numpy())
        outputs = np.array(all_outputs)

    elif model_type == 'mlp':
        model = MLPBaselineImproved(input_size=X.shape[1])
        tensor_val = torch.tensor(X_val).float()
        model.eval()
        with torch.no_grad():
            outputs = model(tensor_val).squeeze().cpu().numpy()
    else:
        raise ValueError("❌ Unsupported model type: Choose 'mlp' or 'gcn'")

    print(f"[{label}] ✅ Output shape: {outputs.shape}")
    save_output(torch.tensor(outputs), out_file)

    print(f"[{label}] 📊 -- Validation Metrics --")
    eval_metrics(outputs, y_val)
    return outputs, y_val

def run_baseline_fold(X_train, y_train, X_val, y_val, label, model_type='mlp'):
    if model_type == 'gcn':
        if X_val.ndim == 3:
            num_samples, N, N2 = X_val.shape
            if N != N2:
                raise ValueError(f"❌ Input matrix must be square (got {N}x{N2})")
            tensor_val = torch.tensor(X_val).float()
        elif X_val.ndim == 2:
            N = int(np.sqrt(X_val.shape[1]))
            if N * N != X_val.shape[1]:
                raise ValueError(f"❌ Invalid shape: {X_val.shape[1]} can't reshape to square.")
            tensor_val = torch.tensor(X_val).float().reshape(-1, N, N)
        else:
            raise ValueError(f"❌ Unsupported shape for GCN: {X_val.shape}")

        model = BrainGCN(in_feats=1, h_feats=16, n_classes=1)
        edge_index = torch.combinations(torch.arange(N), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        batch = torch.zeros(N, dtype=torch.long)
        node_features = tensor_val.mean(dim=2)
        all_outputs = []
        model.eval()
        with torch.no_grad():
            for nf in node_features:
                out = model(nf.unsqueeze(1), edge_index, batch)
                all_outputs.append(out.squeeze().cpu().numpy())
        outputs = np.array(all_outputs)

    elif model_type == 'mlp':
        model = MLPBaselineImproved(input_size=X_train.shape[1])
        tensor_val = torch.tensor(X_val).float()
        model.eval()
        with torch.no_grad():
            outputs = model(tensor_val).squeeze().cpu().numpy()
    else:
        raise ValueError("❌ Unsupported model type: Choose 'mlp' or 'gcn'")

    print(f"[{label}] ✅ Output shape: {outputs.shape}")
    eval_metrics(outputs, y_val)
    return outputs, y_val

def run_kfold_baseline(X_path, y_path, label, model_type='mlp', n_splits=5):
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path).reshape(-1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_outputs, fold_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}/{n_splits} — {label}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        out_fold, y_val_fold = run_baseline_fold(X_train, y_train, X_val, y_val, f"{label} Fold {fold}", model_type)
        fold_outputs.append(out_fold)
        fold_labels.append(y_val_fold)

    all_outputs = np.concatenate(fold_outputs)
    all_labels = np.concatenate(fold_labels)
    print(f"📊Final Averaged Metrics — {label}")
    eval_metrics(all_outputs, all_labels)
    return all_outputs, all_labels

def run_kfold_dast(dfc_path, y_path, label, n_splits=5):
    # --- Load data ---
    X = np.load(dfc_path)  # expected shape: (B, T, N, N)
    y = np.load(y_path).reshape(-1)

    print("Loaded X:", X.shape, "y:", y.shape)

    # --- Align y with subjects ---
    if len(y) != X.shape[0]:
        if len(y) % X.shape[0] == 0:
            repeats = len(y) // X.shape[0]

            # check if labels per subject are identical
            identical = all(len(set(y[i * repeats:(i + 1) * repeats])) == 1 for i in range(X.shape[0]))
            if identical:
                y = y[::repeats]
                print(f"✔ Reduced y to subject-level: {y.shape}")
            else:
                raise ValueError("❌ Labels per subject are not identical. Can't reduce to subject-level.")
        else:
            raise ValueError(f"❌ Length mismatch: expected {X.shape[0]} or {X.shape[0]*X.shape[1]}, got {len(y)}")

    # --- Prepare cross-validator ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_outputs, fold_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(X.shape[0]), y), 1):
        print(f"🔹 Fold {fold}/{n_splits} — {label}")
        print("Train subjects:", train_idx, "Val subjects:", val_idx)

        # select full T slices for each subject
        X_val = X[val_idx]        # shape: (val_B, T, N, N)
        y_val = y[val_idx]        # shape: (val_B,)

        # --- check shape before feeding to model ---
        X_tensor = torch.tensor(X_val, dtype=torch.float32)
        assert X_tensor.dim() == 4, f"❌ Input shape mismatch: got {X_tensor.shape}, expected (B,T,N,N)"

        # --- init model ---
        model = DASTForecaster(n_roi=X_tensor.shape[2])
        model.eval()

        # --- forward pass ---
        with torch.no_grad():
            output = model(X_tensor)

        y_pred = output.cpu().numpy().reshape(-1)

        # --- evaluation ---
        eval_metrics(y_pred, y_val)

        fold_outputs.append(y_pred)
        fold_labels.append(y_val)

    # --- final metrics ---
    all_outputs = np.concatenate(fold_outputs)
    all_labels = np.concatenate(fold_labels)
    print(f"📊 Final Averaged Metrics — {label}")
    eval_metrics(all_outputs, all_labels)
    return all_outputs, all_labels


# ================================
# Run DAST 
# ================================
def run_dast(dfc_path, label, out_file):
    print(f"\n🔹 Running DAST on: {label}")
    X = np.load(dfc_path)  # shape: (B, T, N, N)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    model = DASTForecaster(n_roi=X_tensor.shape[2])
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
    print(f"[{label}] Output shape: {output.shape}")
    save_output(output, out_file)
    return output

# ================================
# Reshape FC to square matrix
# ================================
def reconstruct_square_from_upper(input_path, output_path):
    X = np.load(input_path)

    n_features = X.shape[1]
    N = int((1 + np.sqrt(1 + 8 * n_features)) // 2)
    if N*(N-1)//2 != n_features:
        raise ValueError(f"Cannot reconstruct square: N={N} for features={n_features}")

    X_square = np.zeros((X.shape[0], N, N))

    for i, fc_flat in enumerate(X):
        square = np.zeros((N, N))
        square[np.triu_indices(N, k=1)] = fc_flat
        square += square.T
        X_square[i] = square


    np.save(output_path, X_square)
    print(f"✅ Reconstructed square FC: {output_path}")


def reconstruct_square_from_upper_GCN(input_path, output_path):
    X = np.load(input_path)  # (B, F)
    B, n_features = X.shape
    
    # تخمین N رو به بالا
    N = math.ceil((1 + math.sqrt(1 + 8 * n_features)) / 2)
    print(f"ℹ️ Reconstructing with N={N}, F={n_features}")
    
    # ساخت خروجی
    X_square = np.zeros((B, N, N), dtype=np.float32)
    rows, cols = np.triu_indices(N, k=1)
    
    for i in range(B):
        mat = np.zeros((N, N), dtype=np.float32)
        # فقط به اندازه F داده موجود پر می‌کنیم
        mat[rows[:n_features], cols[:n_features]] = X[i, :n_features]
        mat += mat.T
        X_square[i] = mat
    
    np.save(output_path, X_square)
    print(f"✅ Reconstructed square FC: {output_path}, shape={X_square.shape}")

def reshape_to_square(input_path, output_path):
    X = np.load(input_path)
    n_samples, n_features = X.shape
    N = int(np.sqrt(n_features))
    if N * N != n_features:
        raise ValueError(f"Cannot reshape features={n_features} to square (N×N)")
    X_reshaped = X.reshape(n_samples, N, N)
    np.save(output_path, X_reshaped)
    print(f"✅ Reshaped saved to: {output_path}")

# ================================
# Paths
# ================================
paths = {
    'dfc_resid': 'results/dfc_resid_stack.npy',
    'raw_fc': 'roi_time_series/raw_fc.npy',
    'resid_fc': 'roi_time_series/resid_fc.npy',
    'gcn_raw_ready': 'roi_time_series/raw_fc_square.npy',
    'gcn_resid_ready': 'roi_time_series/resid_fc_square.npy',
    'labels': 'roi_time_series/labels.npy',
    'labels_resid': 'roi_time_series/labels_resid.npy',

}
# ================================
# Run Everything
# ================================
# 1. Run DAST
out_dast = run_dast(paths['dfc_resid'], 'Dynamic Residual FC', 'results/output_dast_resid_fc.npy')
if hasattr(out_dast, "cpu"):
    y_pred = out_dast.cpu().numpy()
else:
    y_pred = np.array(out_dast)
y_pred = y_pred.reshape(-1)  

y_true = np.load(paths['labels']).reshape(-1) 

eval_metrics_dast(y_pred, y_true)


# Run K-Fold MLP
out_mlp_raw, y_val_mlp_raw = run_kfold_baseline(paths['raw_fc'], paths['labels'], 'Raw FC (MLP)', model_type='mlp', n_splits=2)
out_mlp_resid, y_val_mlp_resid = run_kfold_baseline(paths['resid_fc'], paths['labels_resid'], 'Residual FC (MLP)', model_type='mlp', n_splits=2)


# Prepare file for GCN
reconstruct_square_from_upper(paths['raw_fc'], paths['gcn_raw_ready'])
reconstruct_square_from_upper_GCN(paths['resid_fc'], paths['gcn_resid_ready'])

# Run K-Fold GCN
out_gcn_raw, y_val_gcn_raw = run_kfold_baseline(paths['gcn_raw_ready'], paths['labels'], 'Raw FC (GCN)', model_type='gcn', n_splits=2)
out_gcn_resid, y_val_gcn_resid = run_kfold_baseline(paths['gcn_resid_ready'], paths['labels_resid'], 'Residual FC (GCN)', model_type='gcn', n_splits=2)


print("✅ All done.")

