import os
import numpy as np
from Util.config import NIAR, OUTPUT_ROOT, dmn_coords_33, dmn_names_33
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb

# Load features & labels
data = np.load(os.path.join(OUTPUT_ROOT, "dataset", "features_labels.npz"))
X, y = data["X"], data["y"]

# Scale features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Set up Leave-One-Out CV
loo = LeaveOneOut()

# Prepare lists to collect true labels, predictions, and probabilities
y_true, y_pred, y_prob = [], [], []

# Configure an XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

# LOOCV loop
for train_idx, test_idx in loo.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx],      y[test_idx]
    # Train on 27 samples
    xgb_clf.fit(X_tr, y_tr)
    # Predict on the held-out sample
    y_hat = xgb_clf.predict(X_te)[0]
    p_hat = xgb_clf.predict_proba(X_te)[0, 1]
    # Store results
    y_true.append(y_te[0])
    y_pred.append(y_hat)
    y_prob.append(p_hat)

# Compute overall LOOCV metrics
acc_loo = accuracy_score(y_true, y_pred)
auc_loo = roc_auc_score(y_true, y_prob)
cm_loo  = confusion_matrix(y_true, y_pred)

print(f"LOOCV Accuracy: {acc_loo:.3f} ({sum(np.array(y_true)==np.array(y_pred))}/{len(y_true)})")
print(f"LOOCV AUC:      {auc_loo:.3f}")
print("LOOCV Confusion matrix:")
print(cm_loo)

# Retrain on full data to extract feature importances
xgb_full = xgb_clf.fit(X_scaled, y)
importances = xgb_full.feature_importances_
top_idx    = np.argsort(importances)[-10:][::-1]
print("Top 10 features (indices):", top_idx)
print("Importances:", importances[top_idx])
