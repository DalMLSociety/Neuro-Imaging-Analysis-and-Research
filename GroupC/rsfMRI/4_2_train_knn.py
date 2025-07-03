import os
import numpy as np
from Util.config import NIAR, OUTPUT_ROOT
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load features & labels
data = np.load(os.path.join(OUTPUT_ROOT, "dataset", "features_labels.npz"))
X, y = data["X"], data["y"]

# Scale features
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# LOOCV setup
loo = LeaveOneOut()
y_true, y_pred = [], []

# KNN classifier
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

for train_idx, test_idx in loo.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx],      y[test_idx]
    knn.fit(X_tr, y_tr)
    y_hat = knn.predict(X_te)[0]
    y_true.append(y_te[0])
    y_pred.append(y_hat)

# Results
acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
print(f"KNN LOOCV Accuracy: {acc:.3f} ({sum(np.array(y_true)==np.array(y_pred))}/{len(y_true)})")
print("KNN LOOCV Confusion Matrix:")
print(cm)
