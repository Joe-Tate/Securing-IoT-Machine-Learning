"""
Loads UCI HAR's raw windows (Inertial Signals) with their labels, in the same
format as load_group_windows() in preprocessing.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UCI_DIR = PROJECT_ROOT / "data" / "raw" / "uci_har"

# Same signal order as preprocessing.py: total_acc x/y/z, body_acc x/y/z, body_gyro x/y/z
SIGNALS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

# UCI activity names -> our group's labels (sitting, standing and laying all dropped)
LABEL_MAP = {
    "WALKING": "walking",
    "WALKING_UPSTAIRS": "walking_upstairs",
    "WALKING_DOWNSTAIRS": "walking_downstairs",
}


def load_uci_inertial(split="train"):
    """
    Returns X (windows, 128, 9) and a DataFrame with one row of labels per window,
    using the same columns as the group data.
    """
    folder = UCI_DIR / split
    names = pd.read_csv(UCI_DIR / "activity_labels.txt", sep=r"\s+", header=None, index_col=0)[1]
    activity = pd.read_csv(folder / f"y_{split}.txt", header=None)[0].map(names)
    subject = pd.read_csv(folder / f"subject_{split}.txt", header=None)[0]

    X = np.stack([pd.read_csv(folder / "Inertial Signals" / f"{s}_{split}.txt",
                              sep=r"\s+", header=None).to_numpy()
                  for s in SIGNALS], axis=-1).astype(np.float32)

    participant = subject.map(lambda n: f"UCI_{n:02d}")
    meta = pd.DataFrame({
        "activity": activity.map(LABEL_MAP),
        "participant": participant,
        "position": "waist",          # all UCI recordings = phone on the waist
        "platform": "android",
        "recording_id": participant,  # UCI has no recording IDs, so the volunteer is used as an identifier
    })

    keep = meta["activity"].notna().to_numpy()
    return X[keep], meta[keep].reset_index(drop=True)


if __name__ == "__main__":
    for split in ("train", "test"):
        X, meta = load_uci_inertial(split)
        print(f"\n{split.upper()}: {X.shape}, {meta['participant'].nunique()} volunteers")
        print(meta["activity"].value_counts().to_string())