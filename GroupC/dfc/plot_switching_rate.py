import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

df = pd.read_csv("C:/Users/msset/MRI/output/dfc_metrics.csv")

df['group_code'] = df['subject'].str.extract(r'_(c|p)\d+', flags=re.IGNORECASE)[0].str.upper()
df['group'] = df['group_code'].map({'C': 'Control', 'P': 'Patient'})

plot_dir = "C:/Users/msset/MRI/output/plots/switching_rate"
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='group', y='switching_rate', palette='Set2')
sns.stripplot(data=df, x='group', y='switching_rate', color='black', alpha=0.5)
plt.title("Switching Rate Comparison")
plt.ylabel("Switching Rate")
plt.xlabel("Group")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "switching_rate_comparison.png"))
plt.close()

print("✅ Plot saved to:", os.path.join(plot_dir, "switching_rate_comparison.png"))
