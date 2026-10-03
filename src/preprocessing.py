"""""
Turns our collected data into windows in the same format as the UCI HAR dataset:
  1. Split each recording at gaps longer than 0.1 s
  2. Remove noise: median filter + 20 Hz low-pass Butterworth (acc and gyro)
  3. Separate gravity: 0.3 Hz low-pass Butterworth, body_acc = total_acc - gravity
  4. Cut into 128-reading windows (2.56 s) with 50% overlap
"""""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "group_combined.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "group_windows.npz"

FS = 50              # sampling rate (Hz)
WINDOW_SIZE = 128    # readings per window (2.56 s)
WINDOW_STEP = 64     # 50% overlap
GAP_SPLIT_S = 0.1    # start a new segment after a gap longer than this

# 3rd-order low-pass Butterworth filters, as used by UCI HAR
NOISE_FILTER = butter(3, 20, btype="low", fs=FS, output="sos")
GRAVITY_FILTER = butter(3, 0.3, btype="low", fs=FS, output="sos")

LABEL_COLUMNS = ["activity", "participant", "position", "platform", "recording_id"]


def remove_noise(signal):
    """Median filter, then 20 Hz low-pass. signal: (readings, 3) array."""
    signal = np.column_stack([medfilt(axis, 3) for axis in signal.T])
    return sosfiltfilt(NOISE_FILTER, signal, axis=0)


def preprocess_segment(segment):
    """Filter one continuous segment and cut it into (windows, 128, 9)."""
    total_acc = remove_noise(segment[["acc_x", "acc_y", "acc_z"]].to_numpy())
    gyro = remove_noise(segment[["gyro_x", "gyro_y", "gyro_z"]].to_numpy())
    gravity = sosfiltfilt(GRAVITY_FILTER, total_acc, axis=0, padlen=min(len(total_acc) - 1, 150))

    # Same signal order as UCI HAR: total_acc xyz, body_acc xyz, body_gyro xyz
    signals = np.hstack([total_acc, total_acc - gravity, gyro])
    starts = range(0, len(signals) - WINDOW_SIZE + 1, WINDOW_STEP)
    return np.stack([signals[start:start + WINDOW_SIZE] for start in starts])


def build_group_windows():
    df = pd.read_csv(INPUT_PATH)
    windows, labels = [], []

    for _, recording in df.groupby("recording_id", sort=False):
        recording = recording.sort_values("seconds_elapsed")
        segment_id = (recording["seconds_elapsed"].diff() > GAP_SPLIT_S).cumsum()

        for _, segment in recording.groupby(segment_id):
            if len(segment) < WINDOW_SIZE:
                continue  # too short to make a single window
            segment_windows = preprocess_segment(segment)
            windows.append(segment_windows)
            labels += [segment.iloc[0][LABEL_COLUMNS].to_dict()] * len(segment_windows)

    X = np.concatenate(windows).astype(np.float32)
    meta = pd.DataFrame(labels)

    print(f"Windows: {len(X):,}   Shape: {X.shape}")
    print(f"Average total acceleration: {np.linalg.norm(X[:, :, 0:3], axis=2).mean():.3f} g (expect about 1)")
    print(f"Average body acceleration:  {X[:, :, 3:6].mean():.4f} g (expect about 0)")
    print(f"\nWindows per activity:\n{meta['activity'].value_counts().to_string()}")

    np.savez_compressed(OUTPUT_PATH, X=X, **{c: meta[c].to_numpy(dtype=str) for c in LABEL_COLUMNS})
    print(f"\nSaved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return X, meta


def load_group_windows():
    """Returns X (windows, 128, 9) and a DataFrame with one row of labels per window."""
    data = np.load(OUTPUT_PATH)
    return data["X"], pd.DataFrame({c: data[c] for c in LABEL_COLUMNS})


if __name__ == "__main__":
    build_group_windows()