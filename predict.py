"""
predict.py — point this at any image and it tells you cracked or not.

Run train.py first so that crack_model.pt exists.

HOW TO RUN
----------
    python predict.py path/to/photo.jpg          <- one image
    python predict.py path/to/folder/            <- every image in a folder

You can also change the sensitivity:

    python predict.py photo.jpg --threshold 0.3

A LOWER threshold makes the model more eager to say "cracked": it catches
more real cracks but raises more false alarms. For inspection work, missing
a crack is usually worse than a false alarm, so 0.3-0.4 is often sensible.
"""

import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "crack_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"'{MODEL_PATH}' not found. Run train.py first.")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Rebuild the same shape of model, then pour the learned weights back in.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()   # eval() = "stop learning, just answer"

    return model, checkpoint["classes"], checkpoint["img_size"]


def make_transform(img_size):
    # Must match the eval transform used in training — same resize, same
    # normalisation. If these differ, predictions go strange.
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict_one(model, transform, path, threshold):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # add the batch dimension

    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob = torch.sigmoid(logit).item()   # squash to a 0-1 confidence

    return ("cracked" if prob >= threshold else "not_cracked"), prob


def gather_paths(target):
    if os.path.isdir(target):
        return [os.path.join(target, f) for f in sorted(os.listdir(target))
                if f.lower().endswith(IMAGE_EXTS)]
    return [target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="an image file or a folder of images")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="lower = more sensitive to cracks (default 0.5)")
    args = parser.parse_args()

    model, classes, img_size = load_model()
    transform = make_transform(img_size)

    paths = gather_paths(args.target)
    if not paths:
        raise SystemExit(f"No images found at '{args.target}'.")

    print(f"\n{'image':<45} {'verdict':<14} confidence")
    print("-" * 75)

    for path in paths:
        try:
            label, prob = predict_one(model, transform, path, args.threshold)
        except Exception as err:
            print(f"{os.path.basename(path):<45} could not read: {err}")
            continue

        # Confidence in whichever answer was given.
        confidence = prob if label == "cracked" else 1 - prob
        flag = "  <- unsure" if 0.35 < prob < 0.65 else ""
        print(f"{os.path.basename(path):<45} {label:<14} {confidence:.1%}{flag}")

    print()


if __name__ == "__main__":
    main()
