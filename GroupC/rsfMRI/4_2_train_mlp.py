import os
import numpy as np
from Util.config import NIAR, OUTPUT_ROOT
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
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

# MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(15, 4),
    activation='tanh',
    solver='adam',
    alpha=1e-5,
    max_iter=1000,
    random_state=42
)

for train_idx, test_idx in loo.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx],      y[test_idx]
    mlp.fit(X_tr, y_tr)
    y_hat = mlp.predict(X_te)[0]
    p_hat = mlp.predict_proba(X_te)[0,1]
    y_true.append(y_te[0])
    y_pred.append(y_hat)
    y_prob.append(p_hat)

# Results
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
cm  = confusion_matrix(y_true, y_pred)
print(f"MLP LOOCV Accuracy: {acc:.3f} ({sum(np.array(y_true)==np.array(y_pred))}/{len(y_true)})")
print(f"MLP LOOCV AUC:      {auc:.3f}")
print("MLP LOOCV Confusion Matrix:")
print(cm)
