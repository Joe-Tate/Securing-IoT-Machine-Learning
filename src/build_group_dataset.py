import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "group"
MANIFEST_PATH = RAW_DIR / "manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "group_combined.csv"
LOG_PATH = OUTPUT_DIR / "group_build_log.csv"

TRIM_SECONDS = {
    "walking": 4.0,
    "walking_upstairs": 2.5,
    "walking_downstairs": 2.5,
}

EXPECTED_RATE_HZ = 50
RATE_TOLERANCE_HZ = 5
GAP_THRESHOLD_S = 0.05         # report any gap between readings longer than this (mmight be an issue with android)
MERGE_TOLERANCE_MS = 15        # max time difference allowed when pairing acc and gyro readings
INCLUDE_TIMESTAMP = True
TIMEZONE = "Europe/London"

VALID_ACTIVITIES = {"walking", "walking_upstairs", "walking_downstairs"}
VALID_POSITIONS = {"pocket", "waist", "hand"}
MANIFEST_COLUMNS = ["recording_id", "folder", "participant", "activity",
                    "position", "device", "platform", "notes"]

SENSOR_FILES = {
    "ios":     {"acc": "AccelerometerUncalibrated.csv", "gyro": "Gyroscope.csv", "acc_to_g": -1.0},
    "android": {"acc": "AccelerometerUncalibrated.csv", "gyro": "Gyroscope.csv", "acc_to_g": 1 / 9.80665},
}

OUTPUT_COLUMNS = (
    ["recording_id", "participant", "activity", "position", "platform"]
    + (["timestamp_uk"] if INCLUDE_TIMESTAMP else [])
    + ["seconds_elapsed", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
)


# --------------------------------------------------------------------------
# Checking the manifest
# --------------------------------------------------------------------------

def fail(problems):
    """Print every problem found, then stop."""
    print("\nCannot continue. Please fix the following:")
    for problem in problems:
        print(f"  - {problem}")
    print("\nTip: compare folder names against the output of `ls data/raw/group`.")
    sys.exit(1)


def load_manifest():
    """Load manifest.csv and check it thoroughly before any data is touched."""
    if not MANIFEST_PATH.exists():
        fail([f"Manifest not found at {MANIFEST_PATH}"])

    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)
    manifest.columns = manifest.columns.str.strip()

    missing_columns = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing_columns:
        fail([f"Manifest is missing these columns: {missing_columns}"])

    for column in manifest.columns:
        manifest[column] = manifest[column].str.strip()
    manifest["platform"] = manifest["platform"].str.lower()

    problems = []
    for column in ["recording_id", "folder", "participant", "activity", "position", "platform"]:
        for i in manifest.index[manifest[column] == ""]:
            problems.append(f"Manifest line {i + 2}: '{column}' is empty")

    for column in ["recording_id", "folder"]:
        duplicates = manifest.loc[manifest[column].duplicated(), column]
        for value in duplicates:
            problems.append(f"'{value}' appears more than once in the '{column}' column")

    for i, row in manifest.iterrows():
        line = f"Manifest line {i + 2} ({row['recording_id']})"
        if row["activity"] not in VALID_ACTIVITIES:
            problems.append(f"{line}: activity '{row['activity']}' is not one of {sorted(VALID_ACTIVITIES)}")
        if row["position"] not in VALID_POSITIONS:
            problems.append(f"{line}: position '{row['position']}' is not one of {sorted(VALID_POSITIONS)}")
        if row["platform"] not in SENSOR_FILES:
            problems.append(f"{line}: platform '{row['platform']}' is not set up yet "
                            f"(currently supported: {sorted(SENSOR_FILES)})")
        if row["folder"] and not (RAW_DIR / row["folder"]).is_dir():
            problems.append(f"{line}: folder not found: {row['folder']}")

    if problems:
        fail(problems)
    return manifest


# --------------------------------------------------------------------------
# Processing a single recording
# --------------------------------------------------------------------------

def read_metadata(folder_path):
    """Read app settings from Metadata.csv. The device ID is deliberately ignored."""
    path = folder_path / "Metadata.csv"
    if not path.exists():
        return {}, {}

    meta = pd.read_csv(path, dtype=str, keep_default_na=False).iloc[0]
    info = {
        "app_version": meta.get("appVersion", ""),
        "standardisation": meta.get("standardisation", ""),
    }
    sensors = meta.get("sensors", "").split("|")
    rates = meta.get("sampleRateMs", "").split("|")
    return info, dict(zip(sensors, rates))


def load_sensor(path, prefix, scale=1.0):
    """Load one sensor CSV, clean it, and rename x/y/z to e.g. acc_x/acc_y/acc_z."""
    df = pd.read_csv(path)
    missing = {"time", "seconds_elapsed", "x", "y", "z"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    rows_loaded = len(df)
    df = df[["time", "seconds_elapsed", "x", "y", "z"]].dropna()
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    df["time"] = df["time"].astype("int64")

    axes = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    df = df.rename(columns={"x": axes[0], "y": axes[1], "z": axes[2]})
    df[axes] = df[axes] * scale
    return df, rows_loaded - len(df)


def process_recording(row):
    """Clean, merge, trim and label one recording. Returns (data, log, warnings)."""
    folder_path = RAW_DIR / row["folder"]
    files = SENSOR_FILES[row["platform"]]
    log = {"recording_id": row["recording_id"], "activity": row["activity"],
           "position": row["position"]}
    warnings = []

    # Checks the app settings
    info, rates = read_metadata(folder_path)
    log.update(info)
    if not rates:
        warnings.append("No Metadata.csv found, so the app's sampling settings could not be checked")
    for key in ("acc", "gyro"):
        sensor = Path(files[key]).stem
        try:
            interval_ms = float(rates.get(sensor, ""))
            if interval_ms != 1000 / EXPECTED_RATE_HZ:
                warnings.append(f"{sensor} was set to {interval_ms:g} ms between readings "
                                f"(expected {1000 / EXPECTED_RATE_HZ:g} ms = {EXPECTED_RATE_HZ} Hz)")
        except ValueError:
            if rates:
                warnings.append(f"Metadata.csv has no sampling rate for {sensor}")

    # Load both acc and gyro sensors
    for key in ("acc", "gyro"):
        if not (folder_path / files[key]).exists():
            warnings.append(f"Missing file: {files[key]}")
            return None, log, warnings
    acc, log["acc_rows_removed_cleaning"] = load_sensor(folder_path / files["acc"], "acc", files["acc_to_g"])
    gyro, log["gyro_rows_removed_cleaning"] = load_sensor(folder_path / files["gyro"], "gyro")

    # Check the sampling rate and look for gaps
    gaps_between_readings = acc["seconds_elapsed"].diff().dropna()
    measured_rate = 1 / gaps_between_readings.median()
    log["measured_rate_hz"] = round(measured_rate, 1)
    log["gaps_over_50ms"] = int((gaps_between_readings > GAP_THRESHOLD_S).sum())
    if abs(measured_rate - EXPECTED_RATE_HZ) > RATE_TOLERANCE_HZ:
        warnings.append(f"Measured sampling rate is {measured_rate:.1f} Hz (expected {EXPECTED_RATE_HZ} Hz)")
    if log["gaps_over_50ms"]:
        warnings.append(f"{log['gaps_over_50ms']} gap(s) longer than {GAP_THRESHOLD_S * 1000:g} ms "
                        f"(longest {gaps_between_readings.max():.2f} s)")

    # Pair each accelerometer reading with the nearest gyroscope reading
    merged = pd.merge_asof(
        acc, gyro[["time", "gyro_x", "gyro_y", "gyro_z"]],
        on="time", direction="nearest", tolerance=MERGE_TOLERANCE_MS * 1_000_000,
    )
    log["rows_unmatched"] = int(merged[["gyro_x", "gyro_y", "gyro_z"]].isna().any(axis=1).sum())
    merged = merged.dropna()

    # Trim the start and end (removing the extra noise)
    trim = TRIM_SECONDS[row["activity"]]
    start = merged["seconds_elapsed"].min() + trim
    end = merged["seconds_elapsed"].max() - trim
    if end <= start:
        warnings.append(f"Recording is too short to trim {trim:g} s from each end")
        return None, log, warnings
    before = len(merged)
    merged = merged[(merged["seconds_elapsed"] >= start) & (merged["seconds_elapsed"] <= end)]
    log["rows_trimmed"] = before - len(merged)

    # Add labels
    merged = merged.assign(recording_id=row["recording_id"], participant=row["participant"],
                           activity=row["activity"], position=row["position"],
                           platform=row["platform"])

    if INCLUDE_TIMESTAMP:
        merged["timestamp_uk"] = (pd.to_datetime(merged["time"], unit="ns", utc=True)
                                  .dt.tz_convert(TIMEZONE)
                                  .dt.strftime("%Y-%m-%d %H:%M:%S"))
    output = merged[OUTPUT_COLUMNS].reset_index(drop=True)

    # Sanity check: total acceleration should average about 1 g
    magnitude = np.sqrt(output["acc_x"] ** 2 + output["acc_y"] ** 2 + output["acc_z"] ** 2)
    log["median_acc_magnitude_g"] = round(float(magnitude.median()), 3)
    if not 0.8 <= log["median_acc_magnitude_g"] <= 1.2:
        warnings.append(f"Median total acceleration is {log['median_acc_magnitude_g']} g (expected about 1 g). "
                        "Check the units and which accelerometer file is being used.")

    log["final_rows"] = len(output)
    log["minutes"] = round(len(output) / EXPECTED_RATE_HZ / 60, 2)
    return output, log, warnings


# --------------------------------------------------------------------------
# Putting it all together
# --------------------------------------------------------------------------

def print_summary(combined, log_df, skipped):
    def minutes_by(column):
        return (combined.groupby(column).size() / EXPECTED_RATE_HZ / 60).round(1).to_string()

    removed = log_df[["acc_rows_removed_cleaning", "rows_unmatched", "rows_trimmed"]].sum()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Recordings combined: {len(log_df)}   Skipped: {len(skipped)} {skipped if skipped else ''}")
    print(f"Total rows: {len(combined):,}   (~{len(combined) / EXPECTED_RATE_HZ / 60:.1f} minutes)")
    print(f"\nMinutes per activity:\n{minutes_by('activity')}")
    print(f"\nMinutes per position:\n{minutes_by('position')}")
    print(f"\nMinutes per participant:\n{minutes_by('participant')}")
    print("\nRows removed:")
    print(f"  missing values / duplicate timestamps: {removed['acc_rows_removed_cleaning']:,}")
    print(f"  no matching gyroscope reading:         {removed['rows_unmatched']:,}")
    print(f"  start/end trimming:                    {removed['rows_trimmed']:,}")


def build_group_dataset(save=True):
    print(f"Reading manifest: {MANIFEST_PATH}")
    manifest = load_manifest()
    print(f"Manifest OK: {len(manifest)} recordings")

    frames, logs, skipped = [], [], []
    for _, row in manifest.iterrows():
        print(f"\n{row['recording_id']}  ({row['activity']}, {row['position']})")
        data, log, warnings = process_recording(row)
        for warning in warnings:
            print(f"   ! {warning}")
        if data is None:
            print("   SKIPPED")
            skipped.append(row["recording_id"])
            continue
        print(f"   {log['final_rows']:,} rows kept ({log['minutes']} min), "
              f"measured rate {log['measured_rate_hz']} Hz")
        frames.append(data)
        logs.append(log)

    if not frames:
        fail(["No recordings could be processed"])

    combined = pd.concat(frames, ignore_index=True)
    log_df = pd.DataFrame(logs)

    # Everyone should use the same app settings, or the data may not be comparable
    for setting in ("standardisation", "app_version"):
        if setting in log_df and log_df[setting].nunique() > 1:
            print(f"\n! Recordings use different '{setting}' values: {sorted(log_df[setting].unique())}")

    print_summary(combined, log_df, skipped)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        log_df.to_csv(LOG_PATH, index=False)
        print(f"\nSaved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
        print(f"Saved: {LOG_PATH.relative_to(PROJECT_ROOT)}")

    return combined, log_df


if __name__ == "__main__":
    build_group_dataset()