import os
import numpy as np
from Util.config import NIAR, OUTPUT_ROOT
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Load features & labels
data = np.load(os.path.join(OUTPUT_ROOT, "dataset", "features_labels.npz"))
X, y = data["X"], data["y"]

# Scale features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# LOOCV setup
loo = LeaveOneOut()
y_true, y_pred, y_prob = [], [], []

# Logistic Regression with L1 regularization
lr = LogisticRegression(
    penalty='l1',
    solver='saga',
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=10000
)

for train_idx, test_idx in loo.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx],      y[test_idx]
    lr.fit(X_tr, y_tr)
    y_hat = lr.predict(X_te)[0]
    p_hat = lr.predict_proba(X_te)[0,1]
    y_true.append(y_te[0])
    y_pred.append(y_hat)
    y_prob.append(p_hat)

# Results
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
cm  = confusion_matrix(y_true, y_pred)
print(f"Logistic LOOCV Accuracy: {acc:.3f} ({sum(np.array(y_true)==np.array(y_pred))}/{len(y_true)})")
print(f"Logistic LOOCV AUC:      {auc:.3f}")
print("Logistic LOOCV Confusion Matrix:")
print(cm)

# Feature weights
weights = lr.coef_.ravel()
top_idx = np.argsort(np.abs(weights))[-10:][::-1]
print("Top 10 features (indices):", top_idx)
print("Weights:", weights[top_idx])
