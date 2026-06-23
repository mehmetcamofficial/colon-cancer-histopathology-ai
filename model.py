import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = ["cancer", "normal"]
IMG_SIZE = 224


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def load_model(model_path="models/colon_cancer_efficientnet_b0.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def predict(image: Image.Image, model, device):
    transform = get_transform()

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    cancer_prob = float(probabilities[0].cpu())
    normal_prob = float(probabilities[1].cpu())

    predicted_idx = int(torch.argmax(probabilities).cpu())
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx].cpu())

    return {
        "label": predicted_label,
        "confidence": confidence,
        "cancer_probability": cancer_prob,
        "normal_probability": normal_prob,
    }