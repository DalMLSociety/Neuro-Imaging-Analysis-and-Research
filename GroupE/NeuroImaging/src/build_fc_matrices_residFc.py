import numpy as np
import pandas as pd
import os
import math

import numpy as np
import pandas as pd

# مسیر فایل‌ها
labels_path = "roi_time_series/labels.npy"  # فایل لیبل فعلی
fc_path = "roi_time_series/resid_fc.npy"  # داده‌ای که باهاش باید match بشه
out_labels_path = "roi_time_series/labels_aligned.npy"

# 1) لود لیبل‌ها
labels = np.load(labels_path)

# 2) گرفتن تعداد نمونه از FC یا TS
fc_data = np.load(fc_path)
B = fc_data.shape[0]

print(f"📊 Labels shape: {labels.shape}, FC shape: {fc_data.shape}")

# 3) هم‌تراز کردن
if len(labels) > B:
    labels_aligned = labels[:B]  # کوتاه کردن
    print(f"✂️ Labels truncated from {len(labels)} to {B}")
elif len(labels) < B:
    # اگه کوتاهتره، یا باید تکرار کنیم یا با یک مقدار پر کنیم
    reps = (B // len(labels)) + 1
    labels_aligned = np.tile(labels, reps)[:B]
    print(f"➕ Labels repeated from {len(labels)} to {B}")
else:
    labels_aligned = labels
    print("✅ Labels already aligned")

# 4) ذخیره
np.save(out_labels_path, labels_aligned.astype(np.int64))
print(f"💾 Saved aligned labels: {out_labels_path}, shape={labels_aligned.shape}")


exit(0)
csv_path = "results/residual_ts.csv"
out_dir = "roi_time_series"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df = df.dropna(axis=1, how="all")

fc_flat = df.values.astype(np.float32)
np.save(os.path.join(out_dir, "resid_fc.npy"), fc_flat)
print(f"✅ Saved flat resid_fc.npy: {fc_flat.shape}")

F = fc_flat.shape[1]
N = math.ceil((1 + math.sqrt(1 + 8*F)) / 2)  # تخمینی رو به بالا
B = fc_flat.shape[0]
print(f"ℹ️ Deduced N={N} from F={F}")

fc_squares = np.zeros((B, N, N), dtype=np.float32)
rows, cols = np.triu_indices(N, k=1)

for i in range(B):
    mat = np.zeros((N, N), dtype=np.float32)
    # فقط به اندازه F داده موجود پر می‌کنیم
    mat[rows[:F], cols[:F]] = fc_flat[i, :F]
    mat += mat.T
    fc_squares[i] = mat

np.save(os.path.join(out_dir, "resid_fc_square.npy"), fc_squares)
print(f"📐 Saved square resid_fc_square.npy: {fc_squares.shape}")