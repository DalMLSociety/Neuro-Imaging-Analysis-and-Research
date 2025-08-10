import pandas as pd
from scipy.stats import ttest_ind
import re

# Load the metrics CSV
df = pd.read_csv("C:/Users/msset/MRI/output/dfc_metrics.csv")

# Extract group from filename (case-insensitive)
df['group_code'] = df['subject'].str.extract(r'_(c|p)\d+', flags=re.IGNORECASE)[0].str.upper()
df['group'] = df['group_code'].map({'C': 'Control', 'P': 'Patient'})

# Identify all dwell state columns
dwell_cols = [col for col in df.columns if col.startswith("dwell_state_")]

print("T-Test Results (Control vs. Patient):\n")

# Perform a t-test for each state
for col in dwell_cols:
    control_vals = df[df['group'] == 'Control'][col].dropna()
    patient_vals = df[df['group'] == 'Patient'][col].dropna()

    t_stat, p_val = ttest_ind(control_vals, patient_vals, equal_var=False)

    print(f"{col}:")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("  ✅ Significant difference (p < 0.05)")
    print()
