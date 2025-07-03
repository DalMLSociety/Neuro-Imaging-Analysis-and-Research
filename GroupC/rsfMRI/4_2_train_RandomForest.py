import os
import numpy as np
from Util.config import NIAR, OUTPUT_ROOT, dmn_coords_33, dmn_names_33
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Load features & labels
data = np.load(os.path.join(OUTPUT_ROOT, "dataset", "features_labels.npz"))
X, y = data["X"], data["y"]

# Scale features (RF doesn’t strictly need it, but we keep it)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Set up leave-one-out CV
loo = LeaveOneOut()

# We'll collect predictions and probabilities for each left-out sample
y_true, y_pred, y_prob = [], [], []

# Use fixed RF hyperparams—or optionally wrap each fold with a small grid search if you like
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

for train_idx, test_idx in loo.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx],       y[test_idx]
    # Train on 27 samples
    rf.fit(X_tr, y_tr)
    # Predict the single held-out sample
    y_hat = rf.predict(X_te)[0]
    p_hat = rf.predict_proba(X_te)[0,1]
    # Store results
    y_true.append(y_te[0])
    y_pred.append(y_hat)
    y_prob.append(p_hat)

# Compute LOOCV metrics
acc_loo = accuracy_score(y_true, y_pred)
auc_loo = roc_auc_score(y_true, y_prob)
cm_loo  = confusion_matrix(y_true, y_pred)

print(f"LOOCV Accuracy: {acc_loo:.3f} ({sum(np.array(y_true)==np.array(y_pred))}/{len(y_true)})")
print(f"LOOCV AUC:      {auc_loo:.3f}")
print("LOOCV Confusion matrix:")
print(cm_loo)

# Feature importances from model retrained on full data
rf_full = rf.fit(X_scaled, y)
importances = rf_full.feature_importances_
top_idx    = np.argsort(importances)[-10:][::-1]
print("Top 10 features (indices):", top_idx)
print("Importances:", importances[top_idx])
