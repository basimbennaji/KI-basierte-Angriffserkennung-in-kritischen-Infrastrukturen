import os
import joblib
import kagglehub
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================
SAMPLE_RATE = 1.0   # 1.0 = full dataset, 0.1 = 10%
OUTPUT_DIR = "processed_data"
RANDOM_STATE = 42

DROP_COLS = [
    "Timestamp",
    "Dst Port",
    "Flow ID",
    "Source IP",
    "Src IP",
    "Dst IP",
    "Destination IP",
]


def load_and_process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")
    print(f"Dataset path: {dataset_path}")

    csv_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No CSV files found.")

    print("Reading CSV files into memory...")
    dfs = []
    for file in sorted(csv_files):
        print(f"Loading {os.path.basename(file)}")
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    del dfs

    print("Cleaning data...")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if SAMPLE_RATE < 1.0:
        df = df.sample(frac=SAMPLE_RATE, random_state=RANDOM_STATE)

    if "Label" not in df.columns:
        raise ValueError("Column 'Label' not found.")

    print("Mapping labels...")
    df["Label"] = df["Label"].astype(str).str.strip().apply(
        lambda x: 0 if x == "Benign" else 1
    )

    print("Preparing features...")
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.drop(columns=["Label"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    y = df["Label"].astype(np.uint8).copy()

    if X.empty:
        raise ValueError("No numeric features left after preprocessing.")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    del df, X, y

    print("Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Saving...")
    joblib.dump(X_train_scaled.astype(np.float32), os.path.join(OUTPUT_DIR, "X_train.pkl"))
    joblib.dump(X_test_scaled.astype(np.float32), os.path.join(OUTPUT_DIR, "X_test.pkl"))
    joblib.dump(y_train.to_numpy(dtype=np.uint8), os.path.join(OUTPUT_DIR, "y_train.pkl"))
    joblib.dump(y_test.to_numpy(dtype=np.uint8), os.path.join(OUTPUT_DIR, "y_test.pkl"))
    joblib.dump(list(X_train.columns), os.path.join(OUTPUT_DIR, "feature_columns.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump({0: "Benign", 1: "Attack"}, os.path.join(OUTPUT_DIR, "label_mapping.pkl"))

    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Train class distribution:")
    print(pd.Series(y_train).value_counts())
    print("Done.")


if __name__ == "__main__":
    load_and_process()
