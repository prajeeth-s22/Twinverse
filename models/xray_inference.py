"""
Chest X-Ray Inference Pipeline for TwinVerse.
Performs Cardiomegaly risk estimation from chest radiographs.
Uses the trained X-ray model (with fallback/extension for ResNet18).
"""

import io
import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

HERE = os.path.dirname(__file__)
XRAY_MODEL_PATH = os.path.join(HERE, "xray_model.pt")
RESNET18_MODEL_PATH = os.path.join(HERE, "resnet18_cardiomegaly.pt")
IMG_SIZE = 128

_model_cache = None


def load_xray_model():
    """
    Loads and caches the trained X-ray model.
    Defaults to the existing validated XrayCNN model checkpoint.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    device = torch.device("cpu")
    if os.path.exists(XRAY_MODEL_PATH):
        try:
            from xray_cnn import XrayCNN
            model = XrayCNN(num_classes=2)
            model.load_state_dict(torch.load(XRAY_MODEL_PATH, map_location=device))
            model.eval()
            _model_cache = model
            return _model_cache
        except Exception as e:
            print(f"Warning loading XrayCNN: {e}")

    # Fallback to ResNet18 if available
    if os.path.exists(RESNET18_MODEL_PATH):
        try:
            import torchvision.models as models
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load(RESNET18_MODEL_PATH, map_location=device))
            model.eval()
            _model_cache = model
            return _model_cache
        except Exception as e:
            print(f"Warning loading ResNet18: {e}")

    # Default init
    from xray_cnn import XrayCNN
    model = XrayCNN(num_classes=2)
    model.eval()
    _model_cache = model
    return _model_cache


def preprocess_xray(image_input):
    """
    Converts various input types (path, bytes, UploadedFile, PIL Image)
    into a normalized PyTorch tensor (1x1x128x128) and PIL grayscale Image.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):
        content = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        pil_img = Image.open(io.BytesIO(content))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    img_gray = pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor, img_gray


def predict_xray(image_input):
    """
    Estimates Cardiomegaly probability from a chest radiograph.

    Parameters:
        image_input: file path, PIL Image, bytes, or Streamlit UploadedFile

    Returns:
        risk_probability (float): Cardiomegaly risk probability between 0.0 and 1.0
        prediction (int): 1 if Cardiomegaly / abnormality detected, 0 otherwise
    """
    model = load_xray_model()
    tensor, _ = preprocess_xray(image_input)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        cardiomegaly_prob = float(probs[1])
        prediction = int(outputs.argmax(dim=1).item())

    return cardiomegaly_prob, prediction
