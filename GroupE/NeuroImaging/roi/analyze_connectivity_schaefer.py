import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =======================
# Load DMN summary table
# =======================
summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "summary", "dmn_summary.csv"))
df = pd.read_csv(summary_path)

# =======================
# Filter only Schaefer atlas files
# =======================
df = df[df["filepath"].str.contains("schaefer", case=False)]

# =======================
# Prepare correlation matrices
# =======================
group_corrs = {"control": [], "patient": []}
expected_shapes = {"control": None, "patient": None}

for _, row in df.iterrows():
    group = row["group"]
    filepath = row["filepath"]

    try:
        data = pd.read_csv(filepath, header=0)
        data = data.iloc[:, 1:]  # remove first column if it's ROI labels
        data = data.applymap(lambda x: float(x.decode('utf-8')) if isinstance(x, bytes) else float(x))

        corr = data.corr(method="pearson")  # ROI × ROI

        # Check shape consistency
        if expected_shapes[group] is None:
            expected_shapes[group] = corr.shape
        elif corr.shape != expected_shapes[group]:
            print(f" Shape mismatch in {filepath} | Shape: {corr.shape}")
            continue

        group_corrs[group].append(corr.values)

    except Exception as e:
        print(f" Failed to process {filepath}: {e}")

# =======================
# Average correlation matrix per group
# =======================
avg_corr = {}
for group in group_corrs:
    if group_corrs[group]:
        avg_corr[group] = np.mean(group_corrs[group], axis=0)
    else:
        avg_corr[group] = None
        print(f" Warning: No data for group {group}")

# =======================
# Save matrices and plots per group (Schaefer-tagged)
# =======================
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "roi_analysis"))
os.makedirs(output_dir, exist_ok=True)

plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "plots", "roi_analysis_plots"))
os.makedirs(plot_dir, exist_ok=True)

for group in ["control", "patient"]:
    if avg_corr[group] is not None:
        # Save matrix with 'schaefer' tag
        matrix_path = os.path.join(output_dir, f"connectivity_matrix_{group}_schaefer.npy")
        np.save(matrix_path, avg_corr[group])
        print(f" Saved {group} connectivity matrix to: {matrix_path}")

        # Save heatmap with 'schaefer' tag
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_corr[group], cmap="coolwarm", square=True)
        plt.title(f"Average Correlation Matrix ({group.capitalize()} Group) - Schaefer")
        heatmap_path = os.path.join(plot_dir, f"correlation_heatmap_{group}_schaefer.png")
        plt.savefig(heatmap_path)
        print(f" Heatmap saved to: {heatmap_path}")
        plt.close()
