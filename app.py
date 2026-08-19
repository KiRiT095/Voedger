import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import streamlit as st
from streamlit_cropper import st_cropper

MODEL_PATH = "crack_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    return model, checkpoint["classes"], checkpoint["img_size"]


def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def predict(model, transform, image, threshold):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob = torch.sigmoid(logit).item()
    label = "cracked" if prob >= threshold else "not_cracked"
    confidence = prob if label == "cracked" else 1 - prob
    return label, confidence, prob

# UI

st.set_page_config(page_title="Voedger — Crack Detection", page_icon="")
st.title("Voedger — Structural Crack Detection")
st.caption("Upload a structural photo to check it for cracks.")

threshold = st.slider(
    "Sensitivity threshold",
    min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    help="Lower = more sensitive to cracks (catches more, but more false "
         "alarms). 0.3–0.4 is often a sensible choice for inspection work, "
         "since missing a real crack is usually worse than a false alarm.",
)

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

try:
    model, classes, img_size = load_model()
except FileNotFoundError:
    st.error(
        f"Couldn't find '{MODEL_PATH}'. Make sure app.py is in the same "
        f"folder as crack_model.pt, or run train.py first to produce it."
    )
    st.stop()

transform = make_transform(img_size)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    enable_crop = st.checkbox(
        "Crop image before prediction",
        value=False,
        help="Off by default so the model sees the exact same full image "
             "predict.py would. Turn this on to isolate a specific region.",
    )

    if enable_crop:
        ASPECT_RATIOS = {
            "Free": None,
            "1:1 (square)": (1, 1),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "16:9": (16, 9),
            "9:16": (9, 16),
        }
        ratio_label = st.selectbox("Aspect ratio", list(ASPECT_RATIOS.keys()))
        aspect_ratio = ASPECT_RATIOS[ratio_label]

        st.caption(
            "The crop box below does **not** start at the image edges. "
            "Drag its corners out to the full frame unless you specifically "
            "want to trim something out."
        )

        cropped_image = st_cropper(
            image,
            realtime_update=True,
            box_color="#FF4B4B",
            aspect_ratio=aspect_ratio,
            return_type="image",
        )
    else:
        cropped_image = image

    col1, col2 = st.columns(2)

    with col1:
        caption = "Cropped image (used for prediction)" if enable_crop else "Input image"
        st.image(cropped_image, caption=caption, use_container_width=True)
        st.caption(f"Size: {cropped_image.size[0]}×{cropped_image.size[1]} px")

    with col2:
        label, confidence, raw_prob = predict(model, transform, cropped_image, threshold)

        if label == "cracked":
            st.error(f"### Crack detected")
        else:
            st.success(f"### No crack detected")

        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

        if 0.35 < raw_prob < 0.65:
            st.warning("Model is unsure about this one — borderline result.")

        with st.expander("Details"):
            st.write(f"Raw crack probability: {raw_prob:.3f}")
            st.write(f"Threshold used: {threshold}")
            st.write(f"Image resized to: {img_size}×{img_size}")
else:
    st.info("Upload a photo above to run a prediction.")