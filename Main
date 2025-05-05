import streamlit as st
import subprocess
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Streamlit app title
st.title("🎬 Video Screenshot & Tone Analyzer with CLIP")

# Upload video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# Parameters
interval = st.number_input("Frame extraction interval (seconds)", min_value=1, value=10)

# Temporary folders
screenshot_folder = "./screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

# Define text descriptions for tone analysis
descriptions = ["bright and cheerful", "dark and moody", "professional branding", "casual and playful", "serious and formal"]

def analyze_with_clip(image_path, text_descriptions):
    image = Image.open(image_path)
    inputs = processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return {desc: round(prob.item(), 3) for desc, prob in zip(text_descriptions, probs[0])}

if video_file is not None:
    video_path = os.path.join(screenshot_folder, video_file.name)
    
    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    
    st.success(f"Video uploaded: {video_file.name}")
    
    # Extract screenshots button
    if st.button("Extract Screenshots"):
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps=1/{interval}',
            f'{screenshot_folder}/screenshot_%04d.png'
        ]
        subprocess.run(command)
        st.success("Screenshots extracted!")

    # Display extracted screenshots
    screenshots = sorted([f for f in os.listdir(screenshot_folder) if f.endswith(".png")])
    
    if screenshots:
        st.header("Sample Screenshots with Tone Analysis")
        for img_file in screenshots:
            img_path = os.path.join(screenshot_folder, img_file)
            img = Image.open(img_path)
            st.image(img, caption=img_file, use_column_width=True)
            
            # Perform CLIP analysis
            analysis = analyze_with_clip(img_path, descriptions)
            st.write("🧠 *Tone Analysis Results*:")
            st.json(analysis)
    else:
        st.info("No screenshots yet. Click the button above to generate them.")
