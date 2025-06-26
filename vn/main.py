#!/usr/bin/env python3
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, classification_report
)

# ————— EDIT THESE —————
WIN_TRAIN_CSV = r"C:\Users\admin\PycharmProjects\test1\Vietnamese.csv"
WIN_TEST_CSV  = r"C:\Users\admin\PycharmProjects\test1\Vietnamese.csv"
WIN_TRUE_TXT  = r"C:\Users\admin\PycharmProjects\test1\true.txt"
WIN_PRED_TXT  = r"C:\Users\admin\PycharmProjects\test1\predict.txt"

# Your model binary inside WSL
WSL_MODEL     = "/mnt/c/Users/admin/PycharmProjects/vn_model"

MODEL_NAME    = "SentimentNB"  # e.g. your classifier name

# Map label IDs → names
CLASS_NAMES   = ["negative", "neutral", "positive"]
# ————————————————————


def win_to_wsl(win_path: str) -> str:
    r"""Convert C:\path\to\file -> /mnt/c/path/to/file for WSL."""
    drive, rest = win_path.split(":", 1)
    return f"/mnt/{drive.lower()}{rest.replace('\\', '/')}"

def main():
    # 1️⃣ Run model via WSL
    cmd = [
        "wsl",
        WSL_MODEL,
        win_to_wsl(WIN_TRAIN_CSV),
        win_to_wsl(WIN_TEST_CSV),
        win_to_wsl(WIN_TRUE_TXT),
        win_to_wsl(WIN_PRED_TXT)
    ]
    print("→ Running model under WSL:\n  ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"✖ Model failed (exit {e.returncode})")
        return

    # 2️⃣ Load outputs
    try:
        y_true = np.loadtxt(WIN_TRUE_TXT, dtype=int)
        y_pred = np.loadtxt(WIN_PRED_TXT, dtype=int)
    except Exception as e:
        print(f"✖ Error loading labels: {e}")
        return

    if y_true.size == 0 or y_pred.size == 0:
        print("⚠ Label files are empty—aborting.")
        return

    # 3️⃣ Align lengths
    if len(y_true) != len(y_pred):
        n = min(len(y_true), len(y_pred))
        print(f"⚠ Truncating to {n} samples (true={len(y_true)}, pred={len(y_pred)})")
        y_true, y_pred = y_true[:n], y_pred[:n]

    # 4️⃣ Compute metrics
    acc    = accuracy_score(y_true, y_pred)
    w_f1   = f1_score(y_true, y_pred, average='weighted')
    per_f1 = f1_score(y_true, y_pred, average=None)
    fmax   = per_f1.max()

    # 5️⃣ Print report
    print(f"\nTest Accuracy for {MODEL_NAME}: {acc:.2f}")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=2
    ))

    # 6️⃣ Plot confusion matrix
    cm     = confusion_matrix(y_true, y_pred)
    labels = CLASS_NAMES

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        annot_kws={"fontsize":12}
    )
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix for {MODEL_NAME}', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
