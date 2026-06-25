import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = ["cancer", "normal"]
IMG_SIZE = 224


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(model_path="models/colon_cancer_efficientnet_b0.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image: Image.Image):
    transform = get_transform()
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict(image: Image.Image, model, device):
    image_tensor = preprocess_image(image).to(device)

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
        "class_index": predicted_idx,
        "confidence": confidence,
        "cancer_probability": cancer_prob,
        "normal_probability": normal_prob,
    }


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activation
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def create_gradcam_overlay(image: Image.Image, model, device, class_idx=None):
    original = image.convert("RGB")
    original_np = np.array(original)

    input_tensor = preprocess_image(original).to(device)

    if class_idx is None:
        with torch.no_grad():
            output = model(input_tensor)
            class_idx = int(torch.argmax(output, dim=1).item())

    target_layer = model.features[-1]

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor, class_idx)
    gradcam.remove_hooks()

    cam_resized = cv2.resize(
        cam,
        (original_np.shape[1], original_np.shape[0]),
    )

    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.55 * original_np + 0.45 * heatmap)

    return Image.fromarray(overlay), Image.fromarray(heatmap), cam_resized


def create_attention_box_overlay(image: Image.Image, cam_map, threshold=0.55):
    original = image.convert("RGB")
    original_np = np.array(original).copy()

    mask = cam_map > threshold
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return original

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    boxed = original_np.copy()
    thickness = max(2, original_np.shape[1] // 180)

    cv2.rectangle(
        boxed,
        (x1, y1),
        (x2, y2),
        color=(255, 0, 0),
        thickness=thickness,
    )

    return Image.fromarray(boxed)