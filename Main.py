# =============================
# Imports and Global Setup
# =============================
import streamlit as st
import subprocess
import os
import glob
import platform
from pathlib import Path
import shutil
import imageio
import moviepy.config as mpy_config
import numpy as np
from PIL import Image
import openai
import base64

# =============================
# Environment and Dependency Checks
# =============================
# Set the full path to ffmpeg for all subprocesses
# os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
# os.environ["FFMPEG_BINARY"] = "/opt/homebrew/bin/ffmpeg"

# imageio.plugins.ffmpeg.FFMPEG_EXE = "/opt/homebrew/bin/ffmpeg"

# import moviepy.config as mpy_config
# mpy_config.change_settings({"FFMPEG_BINARY": "/opt/homebrew/bin/ffmpeg"})

st.set_page_config(page_title="Video Screenshot & Tone Analyzer", layout="centered")
st.title("🎬 Video Screenshot & Tone Analyzer")

# =============================
# Streamlit UI: Video Upload & Settings
# =============================
# Function to check if a command exists in the system PATH
def command_exists(command):
    """
    Check if a command exists in the system PATH.
    Args:
        command (str): The command to check.
    Returns:
        bool: True if the command exists, False otherwise.
    """
    return shutil.which(command) is not None

# Check for required Python packages
missing_packages = []
try:
    from PIL import Image
except ModuleNotFoundError:
    missing_packages.append("Pillow")

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ModuleNotFoundError as e:
    missing_packages.append(e.name)

if missing_packages:
    with st.error("❌ Missing required libraries:"):
        for package in missing_packages:
            st.write(f"- {package}")
        st.write("Please install missing packages with:")
        st.code(f"pip install {' '.join(missing_packages)}")
    st.stop()

# Find ffmpeg - try multiple approaches
ffmpeg_path = None

# Check common installation paths based on OS
common_paths = []
if platform.system() == "Darwin":  # macOS
    common_paths = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/local/bin/ffmpeg"
    ]
elif platform.system() == "Windows":
    common_paths = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe"
    ]
else:  # Linux and others
    common_paths = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg"
    ]

# First check if ffmpeg is in PATH
if command_exists("ffmpeg"):
    ffmpeg_path = "ffmpeg"  # Use the command name directly
else:
    # Try common paths
    for path in common_paths:
        if os.path.exists(path):
            ffmpeg_path = path
            break

# If not found, allow user to specify path
if not ffmpeg_path:
    st.warning("⚠️ ffmpeg not found in common locations. Please specify the path to ffmpeg.")
    user_path = st.text_input("Path to ffmpeg executable:")
    if user_path and os.path.exists(user_path):
        ffmpeg_path = user_path

# Final check
if not ffmpeg_path:
    st.error("❌ ffmpeg is required but could not be found. Please install ffmpeg or specify its path.")
    with st.expander("Installation instructions"):
        st.markdown("""
        ### Installing ffmpeg:
        
        **macOS:** 
        ```
        brew install ffmpeg
        ```
        
        **Ubuntu/Debian:**
        ```
        sudo apt update
        sudo apt install ffmpeg
        ```
        
        **Windows:**
        1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
        2. Extract to a folder
        3. Add the bin folder to your PATH or specify the full path above
        """)
    st.stop()

if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    try:
        import imageio
        imageio.plugins.ffmpeg.FFMPEG_EXE = ffmpeg_path
    except Exception:
        pass
    try:
        import moviepy.config as mpy_config
        mpy_config.change_settings({"FFMPEG_BINARY": ffmpeg_path})
    except Exception:
        pass

# Create screenshot folder
screenshot_folder = "./screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

# Clean up old screenshots option
##if glob.glob(f"{screenshot_folder}/screenshot_*.png"):
#    if st.checkbox("Clean up previous screenshots"):
#       for file in glob.glob(f"{screenshot_folder}/screenshot_*.png"):
#             os.remove(file)
#        st.success("Previous screenshots removed")

# Main content area (single column)
st.markdown("### Video Player")

# Show a placeholder if no video is loaded
if 'video_file' not in st.session_state or st.session_state['video_file'] is None:
    st.markdown(
        """
        <div style="width:480px; height:270px; border:2px dashed #bbb; display:flex; align-items:center; justify-content:center; color:#bbb; font-size:1.2em; margin-bottom:1em;">
            Video player will appear here
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.video(st.session_state['video_file'], format="video/mp4", start_time=0)

# Controls under the video area
st.header("Controls")
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv", "wmv"])
if video_file is not None:
    st.session_state['video_file'] = video_file
else:
    st.session_state['video_file'] = None

# Checkbox is checked by default
clean_up = st.checkbox("Clean up previous screenshots", value=True)

st.subheader("Extraction Settings")
interval = st.number_input("Frame extraction interval (seconds)", min_value=1, max_value=300, value=10)
max_frames = st.number_input("Maximum frames to extract", min_value=1, max_value=50, value=10)
st.subheader("Analysis Settings")
custom_tone = st.text_input("Add custom tone descriptor:", placeholder="e.g., 'elegant and refined'")

default_descriptions = [
    "bright and cheerful",
    "dark and moody",
    "professional and corporate",
    "casual and playful",
    "serious and formal"
]

if custom_tone:
    if custom_tone not in default_descriptions:
        default_descriptions.append(custom_tone)

selected_tones = st.multiselect(
    "Tone categories to analyze:",
    options=default_descriptions,
    default=default_descriptions[:3]
)

if not selected_tones:
    selected_tones = default_descriptions[:3]  # Default to first 3 if none selected

# =============================
# Image Analysis Functions
# =============================
def get_image_description_gpt4v(image_path):
    """
    Generate a creative storyboard description for an image using GPT-4 Vision.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Storyboard description generated by GPT-4 Vision.
    """
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    prompt = (
        "You are a creative director. "
        "Describe this scene for a storyboard in about 100 words. "
        "Focus on the visual details, mood, and what a viewer should feel or notice."
    )
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

def analyze_image(img_path, selected_tones=None, processor=None, model=None, clip_processor=None, clip_model=None):
    """
    Perform all analyses on a single image using GPT-4 Vision and return results as a dict.
    """
    image = Image.open(img_path)
    # Use GPT-4 Vision for scene/tone description
    description = get_image_description_gpt4v(img_path)
    # Optionally, add basic color/brightness/sharpness using PIL/numpy
    arr = np.array(image.convert('L'))
    brightness = float(np.mean(arr))
    contrast = float(np.std(arr))
    return {
        "description": description,
        "brightness": brightness,
        "contrast": contrast,
        "image": image
    }

# =============================
# Screenshot Extraction & Analysis
# =============================
if video_file is not None and st.button("Extract and Analyze Screenshots"):
    if clean_up:
        # Remove all screenshots
        for file in glob.glob(f"{screenshot_folder}/screenshot_*.png"):
            os.remove(file)
        # Remove all video files in the main directory and screenshots folder
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.wmv", "*.mpeg", "*.mpg"]
        for ext in video_extensions:
            for file in glob.glob(ext):
                os.remove(file)
            for file in glob.glob(os.path.join(screenshot_folder, ext)):
                os.remove(file)
        st.success("Previous screenshots and uploaded video files removed.")
    with st.spinner("Processing video..."):
        try:
            # Save the video file to the screenshot folder
            video_path = os.path.join(screenshot_folder, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            if not os.path.exists(video_path):
                st.error(f"Video file not found at {video_path}")
                st.stop()
            st.write(f"Video path: {video_path}")
            # Get video duration
            duration_cmd = [
                ffmpeg_path, 
                '-i', video_path, 
                '-f', 'null', 
                '-'
            ]
            result = subprocess.run(duration_cmd, stderr=subprocess.PIPE, text=True)
            duration_output = result.stderr
            # Parse duration from ffmpeg output
            duration = None
            for line in duration_output.split('\n'):
                if 'Duration' in line:
                    time_str = line.split('Duration: ')[1].split(',')[0].strip()
                    h, m, s = map(float, time_str.split(':'))
                    duration = h * 3600 + m * 60 + s
                    break
            if duration:
                st.info(f"Video duration: {duration:.2f} seconds")
                # Calculate frame positions
                total_frames = int(duration // interval)
                frames_to_extract = min(total_frames, max_frames)
                # Use ffmpeg to extract frames
                command = [
                    ffmpeg_path,
                    '-i', video_path,
                    '-vf', f'fps=1/{interval}',
                    '-frames:v', str(frames_to_extract),
                    '-q:v', '2',  # Higher quality
                    f'{screenshot_folder}/screenshot_%04d.png'
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"ffmpeg error: {result.stderr}")
                else:
                    st.success(f"Extracted {frames_to_extract} screenshots!")
                    # Load CLIP model for analysis
                    with st.spinner("Loading CLIP model for analysis..."):
                        processor = None
                        screenshots = sorted(glob.glob(f"{screenshot_folder}/screenshot_*.png"))
                        if screenshots:
                            st.header("Screenshot Analysis")
                            all_descriptions = []
                            for idx, img_path in enumerate(screenshots):
                                col_img, col_analysis = st.columns([1, 1])
                                with col_img:
                                    img = Image.open(img_path)
                                    st.image(img, caption=os.path.basename(img_path), use_container_width=True)
                                with col_analysis:
                                    st.subheader(f"Tone Analysis: {os.path.basename(img_path)}")
                                    analysis = analyze_image(img_path, selected_tones, processor, None, None, None)
                                    # Display results
                                    st.info(f"✨ Scene description: **{analysis['description']}**")
                                    st.write(f"**Brightness:** {analysis['brightness']:.2f}")
                                    st.write(f"**Contrast:** {analysis['contrast']:.2f}")
                                    # Generate storyboard description using GPT-4 Vision
                                    try:
                                        description = get_image_description_gpt4v(img_path)
                                        all_descriptions.append((img_path, description))
                                        st.success(f"Scene description generated.")
                                        st.markdown(f"**Storyboard Description:**\n{description}")
                                    except Exception as e:
                                        st.error(f"Failed to generate storyboard description: {str(e)}")
                                    st.markdown("---")
                            # Optionally, display all descriptions at the end
                            st.header("All Storyboard Descriptions")
                            for idx, (img_path, description) in enumerate(all_descriptions):
                                st.image(img_path, caption=f"Scene {idx+1}", use_column_width=True)
                                st.markdown(f"**Scene {idx+1} Description:**")
                                st.info(description)
                                st.markdown("---")
                        else:
                            st.warning("No screenshots were generated.")
            else:
                st.error("Could not determine video duration.")
                st.code(duration_output)  # This will show the ffmpeg output in the Streamlit app
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

# Set your OpenAI API key (use st.secrets in production)
openai.api_key = st.secrets["OPENAI_API_KEY"]
