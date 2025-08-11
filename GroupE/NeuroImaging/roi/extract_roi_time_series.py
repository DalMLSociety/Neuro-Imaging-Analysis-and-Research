import os
import pandas as pd
import numpy as np
import glob

# Define base path to the roi_time_series folder
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "roi_time_series"))
groups = ["control", "patient"]

# Where to save the summary info
output_summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "summary", "dmn_summary.csv"))
os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)

summary_records = []

for group in groups:
    group_path = os.path.join(base_dir, group)
    csv_files = sorted(glob.glob(os.path.join(group_path, "*.csv")))

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        subject = parts[1]  # e.g., C01 or p01
        session = parts[2]  # e.g., rs-1

        try:
            df = pd.read_csv(filepath, header=None)  # shape: [time, ROI]
            n_timepoints, n_rois = df.shape

            summary_records.append({
                "group": group,
                "subject": subject,
                "session": session,
                "filepath": filepath,
                "n_timepoints": n_timepoints,
                "n_rois": n_rois
            })

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Save summary CSV
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(output_summary_path, index=False)
print(f"Saved summary to: {output_summary_path}")
