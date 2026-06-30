"""
Spot the Fake Photo — Screen Recapture Detector
Approach: MobileNetV2 feature extractor + Logistic Regression
0 = real photo, 1 = photo of a screen
"""

import sys
import os
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

# Load MobileNetV2 once at module level
_mobilenet = None

def _get_model():
    global _mobilenet
    if _mobilenet is None:
        m = mobilenet_v2(weights="IMAGENET1K_V1")
        m.classifier = torch.nn.Identity()  # remove final classifier layer
        m.eval()
        _mobilenet = m
    return _mobilenet

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0)
    model = _get_model()
    with torch.no_grad():
        feats = model(tensor)
    return feats.squeeze().numpy()

def predict(image_path: str) -> float:
    feats = extract_features(image_path).reshape(1, -1)
    if not os.path.exists(MODEL_PATH):
        print("WARNING: No model found. Run train.py first.", file=sys.stderr)
        return 0.5
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    prob = data["model"].predict_proba(
        data["scaler"].transform(feats)
    )[0, 1]
    return float(np.clip(prob, 0.0, 1.0))

if __name__ == "__main__":
    print(predict(sys.argv[1]))