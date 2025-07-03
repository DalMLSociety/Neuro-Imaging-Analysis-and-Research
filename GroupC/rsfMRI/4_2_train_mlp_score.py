import os
import numpy as np
import pandas as pd
from Util.config import NIAR, OUTPUT_ROOT
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# Define subject lists
control_ids = [f"C{str(i).zfill(2)}" for i in range(1, 17)]
missing     = {"C03", "C09", "C10", "C13"}  # Subjects with missing data
# Remove missing controls (now 12 controls vs. 16 patients)
control_ids = [sub for sub in control_ids if sub not in missing]
patient_ids = [f"P{str(i).zfill(2)}" for i in range(1, 17)]

# Build the full ID list in the same order you built X,y
all_ids = control_ids + patient_ids

# Load features & labels
data = np.load(os.path.join(OUTPUT_ROOT, "dataset", "features_labels.npz"))
X, y = data["X"], data["y"]

# Scale features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# LOOCV setup
loo = LeaveOneOut()

# Prepare lists to collect for each sample:
rows = []  # each entry: (sub_id, true_label, pred_label, prob_score)

# MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(15,4),
    activation='tanh',
    solver='adam',
    alpha=1e-5,
    max_iter=1000,
    random_state=42
)

for train_idx, test_idx in loo.split(X_scaled):
    sub_id = all_ids[test_idx[0]]
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx],      y[test_idx]
    mlp.fit(X_tr, y_tr)
    y_hat = mlp.predict(X_te)[0]
    p_hat = mlp.predict_proba(X_te)[0,1]
    rows.append((sub_id, int(y_te[0]), int(y_hat), float(p_hat)))

# Assemble into a DataFrame
df_scores = pd.DataFrame(rows, columns=["subject","true_label","pred_label","score"])
# Save to CSV for later
out_csv = os.path.join(OUTPUT_ROOT, "dataset", "mlp_loocv_scores.csv")
df_scores.to_csv(out_csv, index=False)
print(f"Saved per-subject LOOCV scores to {out_csv}")

# Optionally, show overall metrics again
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
acc = accuracy_score(df_scores.true_label, df_scores.pred_label)
auc = roc_auc_score(df_scores.true_label, df_scores.score)
cm  = confusion_matrix(df_scores.true_label, df_scores.pred_label)
print(f"LOOCV Accuracy: {acc:.3f}")
print(f"LOOCV AUC:      {auc:.3f}")
print("LOOCV Confusion matrix:\n", cm)
