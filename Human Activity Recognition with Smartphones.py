# handwritten_recognition.py
# Robust script for training & evaluating a RandomForest on a CSV dataset
# (works for HAR dataset with 'Activity' column or MNIST-like CSV with 'label')

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------- Configure candidate paths --------------------
candidate_paths = [
    r"d:\python_ka_chilla\AI Projects\Human Activity Recognition with Smartphones\train.csv",
    r"d:\python_ka_chilla\AI Projects\Handwritten Digit Recognition with mnist dataset\train.csv",
    "train.csv"
]

# -------------------- Try to locate & load dataset --------------------
df = None
loaded_path = None
for p in candidate_paths:
    try:
        if os.path.isfile(p):
            df = pd.read_csv(p)
            loaded_path = p
            break
    except Exception:
        # continue if path string was problematic
        continue

# fallback: any csv in current dir (if unique)
if df is None:
    cwd = os.getcwd()
    csvs = [f for f in os.listdir(cwd) if f.lower().endswith(".csv")]
    if len(csvs) == 1:
        df = pd.read_csv(csvs[0])
        loaded_path = os.path.join(cwd, csvs[0])
    elif len(csvs) > 1:
        # pick the first, but inform the user
        df = pd.read_csv(csvs[0])
        loaded_path = os.path.join(cwd, csvs[0])
        print(f"Multiple CSVs found in {cwd}; using '{csvs[0]}'.")
    else:
        raise FileNotFoundError(
            "Could not find a 'train.csv' automatically. Put your dataset (train.csv) in the script folder "
            "or update `candidate_paths` above to point to your file."
        )

print(f"Loaded dataset from: {loaded_path}")
print("Preview (first 3 rows):")
print(df.head(3))
print("\nColumns:", df.columns.tolist())

# -------------------- Detect label column --------------------
label_candidates = ['Activity', 'activity', 'label', 'Label', 'target', 'Target', 'class', 'Class']
label_col = None
for c in label_candidates:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    # Heuristic: last column with relatively few unique values is likely the label
    last_col = df.columns[-1]
    unique_count = df[last_col].nunique()
    if unique_count <= 100:  # heuristics: adjust threshold if you expect many classes
        label_col = last_col
        print(f"Guessed label column '{label_col}' (unique values: {unique_count}).")
    else:
        raise KeyError(
            "Could not find a label column automatically. Expected one of: "
            f"{label_candidates}. Columns found: {df.columns.tolist()}"
        )

print(f"Using label column: '{label_col}'")

# -------------------- Prepare X and y --------------------
X = df.drop(columns=[label_col])
y = df[label_col]

# convert any object columns that look numeric
for col in X.select_dtypes(include=['object', 'category']).columns:
    # try convert to numeric
    X[col] = pd.to_numeric(X[col], errors='ignore')

# drop remaining non-numeric features (typical for dataset like HAR: subject id, timestamps etc.)
non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
if non_numeric:
    print(f"Warning: Dropping non-numeric feature columns: {non_numeric}")
    X = X.drop(columns=non_numeric)

# if label is string, encode it
label_encoder = None
label_names = None
if y.dtype == 'object' or str(y.dtype).startswith('category'):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    label_names = list(label_encoder.classes_)
    print("Label encoding applied. Classes:", label_names)
else:
    y = np.array(y)
    # If labels are 0..k-1, optionally set label_names to string versions
    label_names = [str(c) for c in sorted(np.unique(y))]

# -------------------- Handle missing values --------------------
if X.isnull().any().any():
    print("Found missing values in features — filling with column means.")
    X = X.fillna(X.mean())

# -------------------- Scale features --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- Train/test split --------------------
# Use stratify to preserve class distribution
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # fallback if stratify fails (e.g., single class)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

# -------------------- Model training --------------------
print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# -------------------- Prediction & Evaluation --------------------
y_pred = model.predict(X_test)

# accuracy: do NOT pass 'average'
accuracy = accuracy_score(y_test, y_pred)

# choose average for precision/recall/f1
n_classes = len(np.unique(y))
if n_classes == 2:
    avg = 'binary'
else:
    avg = 'macro'

precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

print("\n📊 Evaluation results:")
print(f" - Number of classes: {n_classes}")
print(f" - Accuracy : {accuracy*100:.2f}%")
print(f" - Precision ({avg}): {precision*100:.2f}%")
print(f" - Recall    ({avg}): {recall*100:.2f}%")
print(f" - F1 Score  ({avg}): {f1*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0, target_names=(label_names if len(label_names)==n_classes else None)))

# -------------------- Confusion matrix --------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
if label_names and len(label_names) == cm.shape[0]:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
else:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# -------------------- Optional: save model & scaler --------------------
# To save, uncomment these lines:
# import joblib
# joblib.dump(model, "rf_model.joblib")
# joblib.dump(scaler, "scaler.joblib")

print("\nDone.")
