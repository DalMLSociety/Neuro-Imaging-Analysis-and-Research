import numpy as np
import matplotlib.pyplot as plt
import os

dfc_dir = "C:/Users/msset/MRI/output/dfc_matrices/"
labels = np.load("C:/Users/msset/MRI/output/kmeans_labels.npy", allow_pickle=True)
file_map = np.load("C:/Users/msset/MRI/output/file_map.npy", allow_pickle=True)

for i, seq in enumerate(labels):
    plt.figure(figsize=(12, 1))
    plt.title(f"{file_map[i]}")
    plt.imshow(seq[np.newaxis, :], aspect='auto', cmap='tab10')
    plt.yticks([])
    plt.xlabel("Sliding Windows")
    plt.tight_layout()
    plt.savefig(f"C:/Users/msset/MRI/output/plots/state_timeline/state_timeline_{file_map[i]}.png")
    plt.close()
