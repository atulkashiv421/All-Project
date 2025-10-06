import streamlit as st
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# ===============================
# CONFIG
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model with new syntax
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval().to(device)

# Use official transforms for ResNet18
transform = weights.transforms()

# Class labels
imagenet_classes = weights.meta["categories"]

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="🖼️ AI Image Classifier", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")
topk = st.sidebar.slider("Show Top-K Predictions", 1, 10, 5)

# Main content
st.title("🖼️ AI Image Classifier")
st.markdown(
    "Upload any image from your **gallery/files** and the pretrained "
    "ResNet18 model (trained on ImageNet) will predict what it is."
)

uploaded = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=200)

    # Preprocess
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_p, top_i = torch.topk(probs, topk)

    # Results
    st.subheader("🔮 Predictions")
    for i in range(topk):
        st.write(f"**{imagenet_classes[top_i[i]]}** — {top_p[i].item()*100:.2f}%")
else:
    st.info("👆 Please upload an image to get predictions.")
