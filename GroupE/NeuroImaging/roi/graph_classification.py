import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Construct robust relative path to the CSV file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_dir, 'results', 'roi_analysis', 'graph_features_per_subject_schaefer.csv')

# Load data
df = pd.read_csv(csv_path)

# Prepare features and labels
features = ['global_connectivity', 'graph_density', 'average_degree', 'modularity', 'n_communities']
X = df[features]
y = df['group'].map({'control': 0, 'patient': 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base models
estimators = [
    ('lr', LogisticRegression(max_iter=500)),
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('svm', SVC(kernel='linear', probability=True)),
    ('xgb', XGBClassifier(n_estimators=20, use_label_encoder=False, eval_metric='logloss'))
]

# Stacking
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500),
    passthrough=True
)

# Train and evaluate
stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)
print(classification_report(y_test, y_pred))
