import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# Load features
df = pd.read_csv("F:/dal/NIAR_Project_Zahra/Neuro-Imaging-Analysis-and-Research/GroupE/zahra/results/graph_features_per_subject_schaefer.csv")

# print("Columns:", df.columns.tolist())
# exit()

# Extract group from filename
def infer_group(name):
    name = name.lower()
    if "_p" in name:
        return "patient"
    elif "_c" in name:
        return "control"
    return "unknown"

df["Group"] = df["group"]

# Separate groups
control_df = df[df["Group"] == "control"]
patient_df = df[df["Group"] == "patient"]

# Perform t-tests and collect results
results = []
features = ["global_connectivity", "graph_density", "average_degree", "modularity", "n_communities"]

for feature in features:
    stat, pval = ttest_ind(control_df[feature], patient_df[feature], equal_var=False, nan_policy='omit')
    results.append((feature, stat, pval))

# Print results
print("\n=== Feature Comparison Between Groups ===")
for feature, stat, pval in results:
    sig = " Significant" if pval < 0.05 else "—"
    print(f"{feature:<25} | t = {stat:>6.2f} | p = {pval:.4f} {sig}")

# Optional: plot distributions
plt.figure(figsize=(12, 6))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=df, x="Group", y=feature, palette="Set2")
    plt.title(feature)
    plt.xlabel("")
plt.tight_layout()
plt.show()
