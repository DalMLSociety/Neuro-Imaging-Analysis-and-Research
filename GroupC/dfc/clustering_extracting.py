import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
dfc_dir = "C:/Users/msset/MRI/output/dfc_matrices"
k = 4  # number of brain states (you can tune this later)

# Step 1: Load all dFC vectors and combine
all_vectors = []
subject_windows = []  # track how many windows per subject
file_map = []

for file in sorted(os.listdir(dfc_dir)):
    if file.endswith(".npy"):
        vecs = np.load(os.path.join(dfc_dir, file))  # (n_windows, n_features)
        all_vectors.append(vecs)
        subject_windows.append(len(vecs))
        file_map.append(file)

all_vectors = np.vstack(all_vectors)
print("All dFC vectors combined:", all_vectors.shape)

# Step 2: KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(all_vectors)

# Step 3: Split back into subjects
state_sequences = []
start = 0
for count in subject_windows:
    state_seq = labels[start:start + count]
    state_sequences.append(state_seq)
    start += count

# Step 4: Compute metrics per subject
results = []
for subj, seq in zip(file_map, state_sequences):
    transitions = np.sum(np.diff(seq) != 0)
    switching_rate = transitions / (len(seq) - 1)

    dwell_times = Counter()
    current = seq[0]
    count = 1
    for s in seq[1:]:
        if s == current:
            count += 1
        else:
            dwell_times[current] += count
            current = s
            count = 1
    dwell_times[current] += count  # last one

    # Count number of entries into each state (episodes)
    episode_counts = Counter()
    current = seq[0]
    episode_counts[current] += 1
    for s in seq[1:]:
        if s != current:
            episode_counts[s] += 1
            current = s

    dwell_avg = {state: dwell_times[state] / episode_counts[state] for state in dwell_times}

    results.append({
        "subject": subj,
        "switching_rate": switching_rate,
        "avg_dwell_time": dwell_avg
    })

for r in results:
    print(f"{r['subject']}: switching={r['switching_rate']:.3f}, dwell={r['avg_dwell_time']}")

np.save("C:/Users/msset/MRI/output/kmeans_labels.npy", state_sequences, allow_pickle=True)
np.save("C:/Users/msset/MRI/output/file_map.npy", file_map, allow_pickle=True)

flat_results = []
for r in results:
    row = {
        "subject": r["subject"],
        "switching_rate": r["switching_rate"]
    }
    for state, dwell in r["avg_dwell_time"].items():
        row[f"dwell_state_{state}"] = dwell
    flat_results.append(row)

df = pd.DataFrame(flat_results)
df['group'] = df['subject'].str.extract(r'_(C|P)\d+')[0].map({'C': 'Control', 'P': 'Patient'})
df.to_csv("C:/Users/msset/MRI/output/dfc_metrics.csv", index=False)
print("✅ Saved dfc_metrics.csv with switching rates and dwell times.")
