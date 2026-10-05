"""
Turns windows (windows, 128, 9) into a table of features, one row per window.
The SAME function is used for UCI HAR and the group data, so their features
are directly comparable.

Each window has 128 readings from 9 signals (total_acc, body_acc and body_gyro on x, y, z)

The 7 calculated features:
  time:      mean, std, min, max, energy
  frequency: dominant frequency (step rhythm), spectral energy
"""

import numpy as np
import pandas as pd

FS = 50

# Same order as preprocessing.py and load_uci.py, plus the three magnitudes
CHANNELS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_mag", "body_acc_mag", "body_gyro_mag",
]


def extract_features(X):
    """X: (windows, 128, 9) array. Returns a DataFrame of 84 features per window."""
    # Add the three magnitudes: sqrt(x^2 + y^2 + z^2) for each sensor
    magnitudes = [np.linalg.norm(X[:, :, i:i + 3], axis=2, keepdims=True) for i in (0, 3, 6)]
    signals = np.concatenate([X] + magnitudes, axis=2)          # (windows, 128, 12)

    # Frequency content of each channel (average removed so it shows movement only)
    centred = signals - signals.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centred, axis=1)) ** 2
    freqs = np.fft.rfftfreq(signals.shape[1], d=1 / FS)         # 0 to 25 Hz, steps of ~0.39 Hz

    features = {
        "mean": signals.mean(axis=1),
        "std": signals.std(axis=1),
        "min": signals.min(axis=1),
        "max": signals.max(axis=1),
        "energy": (signals ** 2).mean(axis=1),
        "dominant_freq": freqs[power[:, 1:, :].argmax(axis=1) + 1],   # skip 0 Hz
        "spectral_energy": power.sum(axis=1) / signals.shape[1],
    }

    columns = {f"{channel}_{name}": values[:, i]
               for name, values in features.items()
               for i, channel in enumerate(CHANNELS)}
    return pd.DataFrame(columns)


if __name__ == "__main__":
    # Checks that both datasets give the same feature columns
    from load_uci import load_uci_inertial
    from preprocessing import load_group_windows

    X_uci, _ = load_uci_inertial("train")
    X_group, _ = load_group_windows()
    uci_features, group_features = extract_features(X_uci), extract_features(X_group)
    print(f"UCI train: {uci_features.shape}   Group: {group_features.shape}")
    print(f"Same columns: {list(uci_features.columns) == list(group_features.columns)}")
    print(f"Any missing values: {uci_features.isna().any().any() or group_features.isna().any().any()}")