import os
import numpy as np

control_count = len([f for f in os.listdir('roi_time_series/aal3/control') if f.endswith('.csv')])
patient_count = len([f for f in os.listdir('roi_time_series/aal3/patient') if f.endswith('.csv')])

labels = np.array([0]*control_count + [1]*patient_count)
np.save('roi_time_series/labels.npy', labels)
print("✅ labels.npy generated!")




