import sys

import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def load_model(weights_path="cnn_cifar10.pth"):
    model = SimpleCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def predict(image_path, weights_path="cnn_cifar10.pth"):
    model = load_model(weights_path)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    return CLASSES[predicted_idx], probabilities[predicted_idx].item()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    label, confidence = predict(sys.argv[1])
    print(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")
