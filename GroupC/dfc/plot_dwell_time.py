import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

df = pd.read_csv("C:/Users/msset/MRI/output/dfc_metrics.csv")

df['group_code'] = df['subject'].str.extract(r'_(c|p)\d+', flags=re.IGNORECASE)[0].str.upper()
df['group'] = df['group_code'].map({'C': 'Control', 'P': 'Patient'})

dwell_cols = [col for col in df.columns if col.startswith("dwell_state_")]
dwell_df = df[["subject", "group"] + dwell_cols]

dwell_long = dwell_df.melt(id_vars=['subject', 'group'],
                           var_name='state', value_name='dwell_time')
dwell_long['state'] = dwell_long['state'].str.extract(r'(\d+)').astype(int)

plot_dir = "C:/Users/msset/MRI/output/plots/dwell_time"
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(12, 5))
sns.barplot(data=dwell_long, x='state', y='dwell_time', hue='group', ci='sd', palette='Set2')
plt.title("Average Dwell Time per State by Group")
plt.ylabel("Dwell Time")
plt.xlabel("State")
plt.legend(title='Group')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "dwell_time_by_state.png"))
plt.close()

print("✅ Dwell time plot saved:", os.path.join(plot_dir, "dwell_time_by_state.png"))
