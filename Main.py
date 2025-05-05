# Streamlit app with guidance for ffmpeg installation
import streamlit as st
import subprocess
import os
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ModuleNotFoundError:
    CLIP_AVAILABLE = False

# App title
st.title("🎬 Video Screenshot & Tone Analyzer")

if not CLIP_AVAILABLE:
    st.warning("⚠️ The 'transformers' or 'torch' module is not installed. Please install them with 'pip install transformers torch' to enable tone analysis.")

# Upload video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Parameters
interval = st.number_input("Frame extraction interval (seconds)", min_value=1, value=10)

# Create folder for screenshots
screenshot_folder = "./screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

if CLIP_AVAILABLE:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    descriptions = [
        "bright and cheerful",
        "dark and moody",
        "professional branding",
        "casual and playful",
        "serious and formal"
    ]

    def analyze_with_clip(image_path, text_descriptions):
        image = Image.open(image_path)
        inputs = processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return {desc: round(prob.item(), 3) for desc, prob in zip(text_descriptions, probs[0])}

# ffmpeg check and user guidance
ffmpeg_installed = subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0
if not ffmpeg_installed:
    st.error("❌ ffmpeg is not installed or not found in PATH.")
    st.write("To install ffmpeg:")
    st.code("brew install ffmpeg  # on macOS (Homebrew)")
    st.code("sudo apt-get install ffmpeg  # on Ubuntu/Linux")
    st.code("choco install ffmpeg  # on Windows (Chocolatey)")

if video_file is not None and ffmpeg_installed:
    video_path = os.path.join(screenshot_folder, video_file.name)

    # Save video file
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success(f"Video uploaded: {video_file.name}")

    if st.button("Extract Screenshots"):
        try:
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'fps=1/{interval}',
                f'{screenshot_folder}/screenshot_%04d.png'
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"ffmpeg error: {result.stderr}")
            else:
                st.success("Screenshots extracted!")
        except Exception as e:
            st.error(f"Error running ffmpeg: {e}")

    screenshots = sorted([f for f in os.listdir(screenshot_folder) if f.endswith(".png")])

    if screenshots:
        st.header("Sample Screenshots")
        for img_file in screenshots:
            img_path = os.path.join(screenshot_folder, img_file)
            img = Image.open(img_path)
            st.image(img, caption=img_file, use_column_width=True)

            if CLIP_AVAILABLE:
                analysis = analyze_with_clip(img_path, descriptions)
                st.write("🧠 *Tone Analysis Results*:")
                st.json(analysis)
            else:
                st.info("Install 'transformers' and 'torch' to enable tone analysis.")
    else:
        st.info("No screenshots yet. Click the button above to generate them.")
