"""
results_log.py

Keeps a permanent record of every model run, so old results are never lost
when the data changes.

"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "results" / "results_log.csv"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
ACTIVITIES = ["walking", "walking_upstairs", "walking_downstairs"]


def log_run(model_name, train_data, test_data, y_train, y_test, predictions, notes=""):
    """Add one row to results_log.csv and return a new folder for this run's figures."""
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    per_activity_f1 = f1_score(y_test, predictions, labels=ACTIVITIES, average=None, zero_division=0)

    row = {
        "run_id": run_id,
        "model": model_name,
        "train_data": train_data,
        "test_data": test_data,
        "train_windows": len(y_train),
        "test_windows": len(y_test),
        # Which version of the data this was: how many windows of each activity
        **{f"train_{a}": int((pd.Series(y_train) == a).sum()) for a in ACTIVITIES},
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "macro_f1": round(f1_score(y_test, predictions, average="macro"), 4),
        **{f"f1_{a}": round(score, 4) for a, score in zip(ACTIVITIES, per_activity_f1)},
        "notes": notes,
    }

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)

    run_folder = RUNS_DIR / run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"Logged run {run_id}  (accuracy {row['accuracy']}, macro F1 {row['macro_f1']})")
    return run_folder


def load_log():
    """Every run so far, as a table."""
    return pd.read_csv(LOG_FILE)