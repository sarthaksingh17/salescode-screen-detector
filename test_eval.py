import numpy as np
import os
import pickle
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from predict import _get_model, MODEL_PATH

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def augment_image(img: Image.Image, mode: str) -> Image.Image:
    if mode == "original": return img
    elif mode == "flip": return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif mode == "blur": return img.filter(ImageFilter.GaussianBlur(radius=1.5))
    elif mode == "jpeg_compress":
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif mode == "brightness_up": return ImageEnhance.Brightness(img).enhance(1.25)
    elif mode == "brightness_down": return ImageEnhance.Brightness(img).enhance(0.8)
    return img

AUGMENTATIONS = ["original", "flip", "blur", "jpeg_compress", "brightness_up", "brightness_down"]

def extract_features_from_image(img: Image.Image) -> np.ndarray:
    tensor = _transform(img).unsqueeze(0)
    model = _get_model()
    with torch.no_grad():
        feats = model(tensor)
    return feats.squeeze().numpy()

def load_data():
    X_raw, y_raw = [], []
    X_aug, y_aug, groups_aug = [], [], []
    
    group_id = 0
    print("Extracting features...")
    for label, folder, cls in [("real", "real", 0), ("screen", "screen", 1)]:
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
            path = os.path.join(folder, fname)
            img = Image.open(path).convert("RGB")
            
            # Raw
            feat_raw = extract_features_from_image(img)
            X_raw.append(feat_raw)
            y_raw.append(cls)
            
            # Augmented
            for aug_mode in AUGMENTATIONS:
                aug_img = augment_image(img, aug_mode)
                feat_aug = extract_features_from_image(aug_img)
                X_aug.append(feat_aug)
                y_aug.append(cls)
                groups_aug.append(group_id)
                
            group_id += 1
            
    return np.array(X_raw), np.array(y_raw), np.array(X_aug), np.array(y_aug), np.array(groups_aug)

if __name__ == "__main__":
    X_raw, y_raw, X_aug, y_aug, groups_aug = load_data()
    
    print("\n" + "="*50)
    print("1. Raw Data LOOCV (Baseline)")
    print("="*50)
    scaler_raw = StandardScaler()
    X_raw_scaled = scaler_raw.fit_transform(X_raw)
    
    # Without PCA
    preds_raw = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000, random_state=42), X_raw_scaled, y_raw, cv=LeaveOneOut())
    print(f"LOOCV Accuracy (No PCA): {accuracy_score(y_raw, preds_raw)*100:.1f}%")
    
    # With PCA
    pca_raw = PCA(n_components=0.95, svd_solver="full")
    X_raw_pca = pca_raw.fit_transform(X_raw_scaled)
    preds_raw_pca = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000, random_state=42), X_raw_pca, y_raw, cv=LeaveOneOut())
    print(f"LOOCV Accuracy (With PCA - {pca_raw.n_components_} components): {accuracy_score(y_raw, preds_raw_pca)*100:.1f}%")

    print("\n" + "="*50)
    print("2. Augmented Data Grouped CV (Leakage-Free)")
    print("="*50)
    scaler_aug = StandardScaler()
    X_aug_scaled = scaler_aug.fit_transform(X_aug)
    pca_aug = PCA(n_components=0.95, svd_solver="full")
    X_aug_pca = pca_aug.fit_transform(X_aug_scaled)
    
    cv_grouped = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    
    # We use C=0.05 as found by the grid search earlier
    clf_aug = LogisticRegression(C=0.05, max_iter=2000, random_state=42)
    preds_aug = cross_val_predict(clf_aug, X_aug_pca, y_aug, cv=cv_grouped, groups=groups_aug)
    print(f"10-Fold Grouped CV Accuracy (With PCA - {pca_aug.n_components_} components): {accuracy_score(y_aug, preds_aug)*100:.1f}%")

    print("\n" + "="*50)
    print("3. Training Accuracy (Loaded Model)")
    print("="*50)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        
        feats_scaled = data["scaler"].transform(X_aug)
        if "pca" in data:
            feats_scaled = data["pca"].transform(feats_scaled)
        
        train_preds = data["model"].predict(feats_scaled)
        print(f"Training Accuracy on Augmented Data: {accuracy_score(y_aug, train_preds)*100:.1f}%")
    else:
        print("Model not found.")
