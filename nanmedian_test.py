import sys

from pylab import plot, show
import numpy as np
import scipy.signal

from ap_loader import load_alphapose_json
from pose_tracker import harmonize_indices
from sequence_tools import select_sequence_idx
from recon3d import seq_as_array, drop_low_scores

from nanmedianfilt import nanmedianfilt

fn = sys.argv[1] if len(
    sys.argv) > 1 else '2023-01-18\\Subjects\\S1\\Pose\\S1_02_vt-sync\\alphapose-results.json'
s = load_alphapose_json(fn)
print("Harmonize:")
harmonize_indices(s)
print(f"Loaded '{fn}' (frames: {len(s)})")

poi_sequence = select_sequence_idx(s, 1)
arr = seq_as_array(poi_sequence)
drop_low_scores(arr, 0.5)
print("x nans:", np.isnan(arr[:,::3]).sum(0))
print("y nans:", np.isnan(arr[:,1::3]).sum(0))
print(np.arange(arr.shape[0])[np.isnan(arr[:,0])])

b, a = scipy.signal.butter(4, 16, 'low', fs=240)

k = 3 * 7 + 1
x = arr[:, k]
plot(x)

# simulate NaN issue
# median
y = nanmedianfilt(x, 7)

# 4th order Butterworth LP
y[np.isnan(y)] = 0
y = scipy.signal.filtfilt(b, a, y)

plot(x, color="black", linewidth=0.5)
plot(y, color='red', linewidth=1.5, alpha=0.8)
show()


# filter
for col in range(arr.shape[1] // 3):
    for axis in range(2):
        col_id = 3 * col + axis

        arr[:, col_id] = nanmedianfilt(arr[:, col_id], 7)
        arr[:, col_id] = scipy.signal.filtfilt(b, a, arr[:, col_id])

print("-"*12, "After filtering")
print("x nans:", np.isnan(arr[:,::3]).sum(0))
print("y nans:", np.isnan(arr[:,1::3]).sum(0))

plot(arr)  # arr[:,3*16])
show()
