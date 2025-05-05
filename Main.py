import streamlit as st
import subprocess
import os
import glob
import platform
from pathlib import Path
import shutil
import imageio
import moviepy.config as mpy_config

# Set the full path to ffmpeg for all subprocesses
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
os.environ["FFMPEG_BINARY"] = "/opt/homebrew/bin/ffmpeg"

try:
    import imageio
    imageio.plugins.ffmpeg.download = lambda: None  # Prevents old download attempts
    imageio.plugins.ffmpeg.FFMPEG_EXE = "/opt/homebrew/bin/ffmpeg"
except Exception:
    pass

try:
    import moviepy.config as mpy_config
    mpy_config.change_settings({"FFMPEG_BINARY": "/opt/homebrew/bin/ffmpeg"})
except Exception:
    pass

st.set_page_config(page_title="Video Screenshot & Tone Analyzer", layout="wide")
st.title("🎬 Video Screenshot & Tone Analyzer")

# Function to check if a command exists in the system PATH
def command_exists(command):
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
else:
    st.success(f"✅ Using ffmpeg at: {ffmpeg_path}")

# Create screenshot folder
screenshot_folder = "./screenshots"
os.makedirs(screenshot_folder, exist_ok=True)

# Clean up old screenshots option
if glob.glob(f"{screenshot_folder}/screenshot_*.png"):
    if st.checkbox("Clean up previous screenshots"):
        for file in glob.glob(f"{screenshot_folder}/screenshot_*.png"):
            os.remove(file)
        st.success("Previous screenshots removed")

# App interface
col1, col2 = st.columns([2, 1])

# Main controls
with col2:
    st.header("Controls")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv", "wmv"])
    
    st.subheader("Extraction Settings")
    interval = st.number_input("Frame extraction interval (seconds)", 
                               min_value=1, max_value=300, value=10)
    
    max_frames = st.number_input("Maximum frames to extract", 
                                min_value=1, max_value=50, value=10)
    
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

# Main content area
with col1:
    if video_file is not None:
        video_path = os.path.join(screenshot_folder, video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        
        st.success(f"Video uploaded: {video_file.name}")
        st.video(video_path)
        
        if st.button("Extract and Analyze Screenshots"):
            with st.spinner("Processing video..."):
                try:
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
                                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                                
                                # Analyze screenshots
                                screenshots = sorted(glob.glob(f"{screenshot_folder}/screenshot_*.png"))
                                
                                if screenshots:
                                    st.header("Screenshot Analysis")
                                    
                                    for img_path in screenshots:
                                        col_img, col_analysis = st.columns([1, 1])
                                        
                                        with col_img:
                                            img = Image.open(img_path)
                                            st.image(img, caption=os.path.basename(img_path), use_column_width=True)
                                        
                                        with col_analysis:
                                            st.subheader(f"Tone Analysis: {os.path.basename(img_path)}")
                                            
                                            # Analyze image with CLIP
                                            image = Image.open(img_path)
                                            inputs = processor(text=selected_tones, images=image, return_tensors="pt", padding=True)
                                            outputs = model(**inputs)
                                            logits_per_image = outputs.logits_per_image
                                            probs = logits_per_image.softmax(dim=1)
                                            
                                            # Create analysis results
                                            results = {}
                                            for tone, prob in zip(selected_tones, probs[0]):
                                                results[tone] = round(float(prob) * 100, 1)  # Convert to percentage
                                            
                                            # Sort by probability
                                            sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
                                            
                                            # Display results
                                            for tone, percentage in sorted_results.items():
                                                st.write(f"{tone}: {percentage}%")
                                                st.progress(percentage/100)
                                            
                                            # Show dominant tone
                                            dominant_tone = max(results, key=results.get)
                                            st.info(f"✨ Dominant tone: **{dominant_tone}**")
                                else:
                                    st.warning("No screenshots were generated.")
                    else:
                        st.error("Could not determine video duration.")
                        
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
    else:
        st.info("Please upload a video file to begin.")
        
        with st.expander("About this app"):
            st.markdown("""
            ## Video Screenshot & Tone Analyzer
            
            This app helps you analyze the visual tone of your videos by:
            
            1. Extracting screenshots at regular intervals
            2. Using CLIP (Contrastive Language-Image Pre-Training) to analyze each frame
            3. Identifying the dominant visual tone in each screenshot
            
            ### Use cases:
            - Analyze brand consistency in marketing videos
            - Evaluate the emotional journey of a film
            - Ensure visual tone matches intended messaging
            - Identify key moments for thumbnails
            
            ### Requirements:
            - ffmpeg must be installed
            - Python packages: streamlit, Pillow, transformers, torch
            """)
