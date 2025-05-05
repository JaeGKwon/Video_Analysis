import streamlit as st
import subprocess
import os
import glob

# Check all required Python libraries at the very start
missing_libraries = []

try:
    from PIL import Image
except ModuleNotFoundError:
    missing_libraries.append('Pillow')

try:
    from transformers import CLIPProcessor, CLIPModel
except ModuleNotFoundError:
    missing_libraries.append('transformers')

try:
    import torch
except ModuleNotFoundError:
    missing_libraries.append('torch')

if missing_libraries:
    st.error(f"❌ Missing required libraries: {', '.join(missing_libraries)}. Please install them using pip.")
    st.stop()

st.title("🎬 Video Screenshot & Tone Analyzer")

screenshot_folder = "./screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
interval = st.number_input("Frame extraction interval (seconds)", min_value=1, value=10)

# Check ffmpeg availability
try:
    ffmpeg_check = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    ffmpeg_installed = ffmpeg_check.returncode == 0
except Exception as e:
    st.warning(f"⚠️ Could not verify ffmpeg installation: {e}")
    ffmpeg_installed = True  # allow to proceed anyway

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

if video_file is not None:
    video_path = os.path.join(screenshot_folder, video_file.name)

    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success(f"Video uploaded: {video_file.name}")

    if not ffmpeg_installed:
        st.warning("⚠️ ffmpeg not detected. Please ensure it is installed and accessible in PATH.")

    if st.button("Extract and Analyze Screenshots"):
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

                screenshots = sorted(glob.glob(f"{screenshot_folder}/screenshot_*.png"))

                if screenshots:
                    st.header("Sample Screenshots")
                    for img_path in screenshots:
                        img = Image.open(img_path)
                        st.image(img, caption=os.path.basename(img_path), use_column_width=True)

                        analysis = analyze_with_clip(img_path, descriptions)
                        st.write(f"🧠 *Tone Analysis for {os.path.basename(img_path)}*:")
                        st.json(analysis)
                else:
                    st.info("No screenshots were generated.")
        except Exception as e:
            st.error(f"Error running ffmpeg: {e}")
else:
    st.info("Please upload a video file to begin.")
