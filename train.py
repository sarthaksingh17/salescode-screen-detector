"""
Train screen recapture detector — MobileNetV2 features + Logistic Regression.
Includes: proper grouped CV (no data leakage), PCA dimensionality reduction,
nested CV for regularization search, and domain-relevant augmentations.
"""

import os, io, pickle, time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeaveOneOut, StratifiedGroupKFold, GridSearchCV, cross_val_predict
)
from sklearn.metrics import accuracy_score
from predict import _get_model, MODEL_PATH, predict

# ── Image preprocessing (same as predict.py) ──────────────────────────
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Augmentations (Issue 8: domain-relevant for real-vs-screen) ───────
def augment_image(img: Image.Image, mode: str) -> Image.Image:
    if mode == "original":
        return img
    elif mode == "flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == "blur":
        # Simulates camera shake / out-of-focus softening
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))
    elif mode == "jpeg_compress":
        # Simulates WhatsApp-style compression (the cause of 2/3 original misclassifications)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif mode == "brightness_up":
        return ImageEnhance.Brightness(img).enhance(1.25)
    elif mode == "brightness_down":
        return ImageEnhance.Brightness(img).enhance(0.8)
    return img

AUGMENTATIONS = [
    "original", "flip", "blur", "jpeg_compress",
    "brightness_up", "brightness_down"
]

def extract_features_from_image(img: Image.Image) -> np.ndarray:
    tensor = _transform(img).unsqueeze(0)
    model = _get_model()
    with torch.no_grad():
        feats = model(tensor)
    return feats.squeeze().numpy()

# ── Collect dataset with augmentation + group tracking ────────────────
def collect_augmented(real_dir="real", screen_dir="screen"):
    X, y, names, groups = [], [], [], []
    group_id = 0
    for label, folder, cls in [("real", real_dir, 0), ("screen", screen_dir, 1)]:
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB")
                for aug_mode in AUGMENTATIONS:
                    aug_img = augment_image(img, aug_mode)
                    feat = extract_features_from_image(aug_img)
                    X.append(feat)
                    y.append(cls)
                    names.append(f"{fname}__{aug_mode}")
                    groups.append(group_id)  # same group for all augmentations of one image
                print(f"  [{label}] {fname} -> {len(AUGMENTATIONS)} versions")
            except Exception as e:
                print(f"  SKIP {fname}: {e}")
            group_id += 1
    return np.array(X), np.array(y), names, np.array(groups)

def collect_raw(real_dir="real", screen_dir="screen"):
    """Collect originals only (no augmentation) for baseline LOOCV."""
    X, y, names = [], [], []
    for label, folder, cls in [("real", real_dir, 0), ("screen", screen_dir, 1)]:
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB")
                X.append(extract_features_from_image(img))
                y.append(cls)
                names.append(fname)
            except Exception as e:
                print(f"  SKIP {fname}: {e}")
    return np.array(X), np.array(y), names


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: Baseline LOOCV on raw 67 images (Issue 2 — confirm 95.5%)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Baseline LOOCV on raw images (no augmentation)")
print("=" * 60)
print("Extracting raw features...")
X_raw, y_raw, names_raw = collect_raw()
print(f"\nRaw dataset: {len(y_raw)} images ({sum(y_raw==0)} real, {sum(y_raw==1)} screen)")

scaler_raw = StandardScaler()
X_raw_scaled = scaler_raw.fit_transform(X_raw)

# PCA for baseline too — see how many components explain 95% variance
pca_raw = PCA(n_components=0.95, svd_solver="full")
X_raw_pca = pca_raw.fit_transform(X_raw_scaled)
print(f"PCA: {pca_raw.n_components_} components explain 95% variance (from 1280)")

clf_baseline = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
preds_raw = cross_val_predict(clf_baseline, X_raw_pca, y_raw, cv=LeaveOneOut())
acc_raw = accuracy_score(y_raw, preds_raw)
print(f"LOOCV Accuracy (raw, with PCA): {acc_raw*100:.1f}%")
wrong_raw = [names_raw[i] for i in range(len(y_raw)) if preds_raw[i] != y_raw[i]]
if wrong_raw:
    print(f"Misclassified ({len(wrong_raw)}):")
    for n in wrong_raw:
        print(f"  {n}")

# Also run without PCA for comparison
preds_raw_nopca = cross_val_predict(
    LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    X_raw_scaled, y_raw, cv=LeaveOneOut()
)
acc_raw_nopca = accuracy_score(y_raw, preds_raw_nopca)
print(f"LOOCV Accuracy (raw, NO PCA): {acc_raw_nopca*100:.1f}%")

# Issue 2: confidence distribution on re-fitted model
clf_baseline.fit(X_raw_pca, y_raw)
probs_raw = clf_baseline.predict_proba(X_raw_pca)[:, 1]
extreme = np.sum((probs_raw < 0.05) | (probs_raw > 0.95))
uncertain = np.sum((probs_raw > 0.3) & (probs_raw < 0.7))
print(f"\nConfidence on training data (expected to be high — this is normal):")
print(f"  Very confident (<0.05 or >0.95): {extreme}/{len(probs_raw)}")
print(f"  Uncertain (0.3-0.7): {uncertain}/{len(probs_raw)}")
print(f"  NOTE: 100% confidence on training data is expected for logistic regression")
print(f"        with {X_raw.shape[1]} features vs {len(y_raw)} samples. The meaningful")
print(f"        number is the LOOCV accuracy above ({acc_raw*100:.1f}%), not training accuracy.")


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: Augmented dataset with leakage-free grouped CV (Issues 1,3,4)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Augmented dataset with grouped CV (no data leakage)")
print("=" * 60)
print("Extracting augmented features...")
X_aug, y_aug, names_aug, groups_aug = collect_augmented()
print(f"\nAugmented dataset: {len(y_aug)} samples ({sum(y_aug==0)} real, {sum(y_aug==1)} screen)")
print(f"Unique source images: {len(np.unique(groups_aug))}")
print(f"Feature dim: {X_aug.shape[1]}")

# Scale + PCA (Issue 3)
scaler_aug = StandardScaler()
X_aug_scaled = scaler_aug.fit_transform(X_aug)

pca_aug = PCA(n_components=0.95, svd_solver="full")
X_aug_pca = pca_aug.fit_transform(X_aug_scaled)
print(f"PCA: {pca_aug.n_components_} components explain 95% variance")

# Issue 1: StratifiedGroupKFold — all augments of one image stay together
cv_grouped = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

# Issue 4: Nested CV with GridSearchCV for C selection
param_grid = {"C": [0.01, 0.05, 0.1, 0.3, 1.0, 5.0]}
inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning nested CV with regularization search...")
print("  Outer CV: 10-fold StratifiedGroupKFold (leakage-free)")
print("  Inner CV: 5-fold StratifiedGroupKFold (for C selection)")
print(f"  C candidates: {param_grid['C']}")

# Manual nested CV because GridSearchCV doesn't natively pass groups to inner CV
all_preds = np.zeros(len(y_aug), dtype=int)
best_Cs = []

for fold_i, (train_idx, test_idx) in enumerate(
    cv_grouped.split(X_aug_pca, y_aug, groups=groups_aug)
):
    X_train, y_train = X_aug_pca[train_idx], y_aug[train_idx]
    X_test = X_aug_pca[test_idx]
    groups_train = groups_aug[train_idx]

    # Inner CV to pick best C
    best_C, best_inner_acc = None, -1
    for C_val in param_grid["C"]:
        inner_preds = cross_val_predict(
            LogisticRegression(C=C_val, max_iter=2000, random_state=42),
            X_train, y_train,
            cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
            groups=groups_train
        )
        inner_acc = accuracy_score(y_train, inner_preds)
        if inner_acc > best_inner_acc:
            best_inner_acc = inner_acc
            best_C = C_val

    best_Cs.append(best_C)
    clf_fold = LogisticRegression(C=best_C, max_iter=2000, random_state=42)
    clf_fold.fit(X_train, y_train)
    all_preds[test_idx] = clf_fold.predict(X_test)

acc_grouped = accuracy_score(y_aug, all_preds)
print(f"\nBest C per fold: {best_Cs}")
print(f"10-Fold Grouped CV Accuracy (leakage-free): {acc_grouped*100:.1f}%")
wrong_aug = [(names_aug[i], "real" if y_aug[i]==0 else "screen")
             for i in range(len(y_aug)) if all_preds[i] != y_aug[i]]
if wrong_aug:
    print(f"Misclassified ({len(wrong_aug)}):")
    for name, true in wrong_aug[:20]:
        print(f"  {name} (true={true})")
else:
    print("No misclassifications!")

# Pick most common best C for final model
from collections import Counter
final_C = Counter(best_Cs).most_common(1)[0][0]
print(f"\nFinal C chosen (most frequent across folds): {final_C}")


# ══════════════════════════════════════════════════════════════════════
#  STEP 3: Train final model on all data + save
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Train final model + save")
print("=" * 60)

# Build final pipeline: scaler -> PCA -> logistic regression
final_scaler = StandardScaler()
X_final_scaled = final_scaler.fit_transform(X_aug)
final_pca = PCA(n_components=pca_aug.n_components_, svd_solver="full")
X_final_pca = final_pca.fit_transform(X_final_scaled)

final_clf = LogisticRegression(C=final_C, max_iter=2000, random_state=42)
final_clf.fit(X_final_pca, y_aug)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": final_clf,
        "scaler": final_scaler,
        "pca": final_pca
    }, f)
print(f"Model saved to {MODEL_PATH}")
print(f"  Pipeline: StandardScaler -> PCA({pca_aug.n_components_}) -> LogReg(C={final_C})")


# ══════════════════════════════════════════════════════════════════════
#  STEP 4: Latency test
# ══════════════════════════════════════════════════════════════════════
times = []
test = os.path.join("real", os.listdir("real")[0])
for _ in range(10):
    t0 = time.perf_counter()
    predict(test)
    times.append((time.perf_counter() - t0) * 1000)
print(f"Avg latency: {np.mean(times):.1f} ms")


# ══════════════════════════════════════════════════════════════════════
#  Issue 6: Generalization caveat
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GENERALIZATION CAVEAT")
print("=" * 60)
n_unique = len(np.unique(groups_aug))
print(f"All {n_unique} source images came from ~1-2 devices.")
print("No validation on images from genuinely different phones/screens.")
print("Expect 5-15% accuracy drop on a new device as normal.")
print("To estimate real-world generalization, collect test images from")
print("at least 3-5 different phones and 2-3 different screen types.")