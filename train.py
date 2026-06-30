"""
Extract MobileNetV2 features from real/ and screen/ folders,
train Logistic Regression, evaluate with Leave-One-Out CV, save model.
"""

import os
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score
from predict import extract_features, MODEL_PATH, predict

def collect(real_dir="real", screen_dir="screen"):
    X, y, names = [], [], []
    for label, folder, cls in [("real", real_dir, 0), ("screen", screen_dir, 1)]:
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            try:
                X.append(extract_features(path))
                y.append(cls)
                names.append(fname)
                print(f"  [{label}] {fname}")
            except Exception as e:
                print(f"  SKIP {fname}: {e}")
    return np.array(X), np.array(y), names

print("Extracting features (this takes ~2-3 mins)...")
X, y, names = collect()
print(f"\nDataset: {len(y)} images ({sum(y==0)} real, {sum(y==1)} screen)")
print(f"Feature dim: {X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
preds = cross_val_predict(clf, X_scaled, y, cv=LeaveOneOut())
acc = accuracy_score(y, preds)
print(f"\nLOOCV Accuracy: {acc*100:.1f}%")

# Show misclassified
wrong = [(names[i], "real" if y[i]==0 else "screen") 
         for i in range(len(y)) if preds[i] != y[i]]
if wrong:
    print(f"Misclassified ({len(wrong)}):")
    for name, true in wrong:
        print(f"  {name} (true={true})")

# Train on all data and save
clf.fit(X_scaled, y)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": clf, "scaler": scaler}, f)
print(f"\nModel saved to {MODEL_PATH}")

# Latency
times = []
import time
test = os.path.join("real", os.listdir("real")[0])
for _ in range(10):
    t0 = time.perf_counter()
    predict(test)
    times.append((time.perf_counter()-t0)*1000)
print(f"Avg latency: {np.mean(times):.1f} ms")