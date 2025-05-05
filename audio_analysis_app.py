import streamlit as st
import os
from Audio_Analysis import AdvancedAudioAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip
import tempfile
import time
from datetime import datetime
import json
import shutil
import openai

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

# Set OpenAI API key from Streamlit secrets (support both formats)
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
elif "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    openai.api_key = st.secrets["openai"]["api_key"]
else:
    st.error("OpenAI API key not found in Streamlit secrets. Please add it as 'OPENAI_API_KEY' or in the [openai] section.")

def generate_interpretation(report):
    try:
        prompt = (
            "Given the following audio analysis results, provide a concise interpretation for a non-technical user. "
            "Evaluate the audio from a user perspective (e.g., 'the music was too intense', 'the narration was clear', etc.). "
            "Also, provide at least one suggestion for what could be improved in the audio to enhance user experience. "
            "\nResults:\n"
            f"{report}\n"
            "\nFormat your answer as follows:\n"
            "User Perspective: <your summary>\n"
            "What Can Be Done Better: <your suggestion>"
        )
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating interpretation: {e}"

# Page configuration
st.set_page_config(
    page_title="Audio Analysis Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'audio_file' not in st.session_state:
    st.session_state['audio_file'] = None
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = None
if 'analysis_report' not in st.session_state:
    st.session_state['analysis_report'] = None
if 'extracted_audio_path' not in st.session_state:
    st.session_state['extracted_audio_path'] = None
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = []

# Sidebar
with st.sidebar:
    st.title("🎵 Audio Analysis")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool analyzes audio from videos to provide insights about:
    - Emotional response
    - Engagement patterns
    - Brand identity
    - Psychological impact
    - Audience reception
    """)
    
    st.markdown("### Supported Formats")
    st.markdown("""
    - Video: MP4, MOV, AVI, MKV, WMV
    - Audio: WAV (extracted from video)
    """)
    
    if st.session_state['analysis_history']:
        st.markdown("### Recent Analyses")
        for i, history in enumerate(st.session_state['analysis_history'][-5:]):
            st.markdown(f"**{i+1}. {history['timestamp']}**")
            st.markdown(f"- File: {history['filename']}")
            st.markdown(f"- Analysis: {', '.join(history['analysis_types'])}")

# Main content
st.title("🎵 Audio Analysis Dashboard")

# File uploader with drag and drop support
st.header("Upload Video")
video_file = st.file_uploader(
    "Drag and drop a video file here or click to browse",
    type=["mp4", "mov", "avi", "mkv", "wmv"],
    help="Upload a video file to analyze its audio content"
)

def extract_audio_from_video(video_path, output_path):
    """Extract audio from video file with progress tracking"""
    try:
        with st.spinner("Extracting audio from video..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize video
            video = VideoFileClip(video_path)
            duration = video.duration
            
            # Extract audio
            audio = video.audio
            audio.write_audiofile(
                output_path,
                logger=None
            )
            
            video.close()
            progress_bar.progress(1.0)
            status_text.text("Audio extraction complete!")
            time.sleep(1)  # Show completion briefly
            progress_bar.empty()
            status_text.empty()
            
            return True
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return False

if video_file is not None:
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, f"temp_{video_file.name}")
        temp_audio_path = os.path.join(temp_dir, f"temp_audio_{os.path.splitext(video_file.name)[0]}.wav")
        
        # Save the uploaded file
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        
        # Extract audio from video
        if extract_audio_from_video(temp_video_path, temp_audio_path):
            # Create a permanent copy of the audio file
            permanent_audio_path = f"audio_{os.path.splitext(video_file.name)[0]}.wav"
            shutil.copy2(temp_audio_path, permanent_audio_path)
            
            st.session_state['audio_file'] = permanent_audio_path
            st.session_state['extracted_audio_path'] = permanent_audio_path
            
            # Display audio player with waveform
            st.audio(permanent_audio_path, format="audio/wav")
            
            # Show file information
            with st.expander("File Information"):
                video = VideoFileClip(temp_video_path)
                st.write(f"Duration: {video.duration:.2f} seconds")
                st.write(f"Resolution: {video.size[0]}x{video.size[1]}")
                st.write(f"FPS: {video.fps}")
                video.close()

# Analysis Options
st.header("Analysis Options")
analysis_types = st.multiselect(
    "Select analysis types:",
    ["Emotional Response", "Engagement Patterns", "Brand Identity", 
     "Psychological Impact", "Audience Reception"],
    default=["Emotional Response", "Engagement Patterns", "Brand Identity", 
             "Psychological Impact", "Audience Reception"],
    help="Choose which aspects of the audio to analyze"
)

# Reference audio for brand identity analysis
if "Brand Identity" in analysis_types:
    ref_path = st.session_state.get('extracted_audio_path')

# Demographic data for audience reception analysis
if "Audience Reception" in analysis_types:
    col1, col2 = st.columns(2)
    with col1:
        demographic_data = {
            "age_group": st.selectbox(
                "Target Age Group",
                ["18-24", "25-34", "35-44", "45-54", "55+"],
                index=1
            ),
            "gender": st.multiselect(
                "Target Gender",
                ["Male", "Female", "Other"],
                default=["Male", "Female"]
            )
        }
    with col2:
        demographic_data["region"] = st.text_input("Target Region", "US")
        demographic_data["language"] = st.selectbox(
            "Primary Language",
            ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Other"]
        )
else:
    demographic_data = None

# Analysis Button
if st.button("Run Analysis", type="primary") and st.session_state['audio_file']:
    with st.spinner("Analyzing audio..."):
        try:
            # Initialize analyzer with the correct file path
            audio_path = st.session_state['audio_file']
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found at {audio_path}")
                
            analyzer = AdvancedAudioAnalyzer(audio_path)
            st.session_state['analyzer'] = analyzer
            
            # Run selected analyses
            report = {}
            
            # Create progress bar for analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run each analysis with progress updates
            total_analyses = len(analysis_types)
            for i, analysis_type in enumerate(analysis_types):
                status_text.text(f"Running {analysis_type} analysis...")
                progress_bar.progress((i + 1) / total_analyses)
                
                if analysis_type == "Emotional Response":
                    report['emotional_response'] = analyzer.emotional_response_analysis()
                elif analysis_type == "Engagement Patterns":
                    report['engagement_patterns'] = analyzer.engagement_pattern_analysis()
                elif analysis_type == "Brand Identity":
                    report['brand_identity'] = analyzer.brand_identity_analysis(ref_path)
                elif analysis_type == "Psychological Impact":
                    report['psychological_impact'] = analyzer.psychological_impact_analysis()
                elif analysis_type == "Audience Reception":
                    report['audience_reception'] = analyzer.audience_reception_analysis(demographic_data)
            
            st.session_state['analysis_report'] = report
            
            # Automatically generate interpretation after analysis
            with st.spinner("Generating interpretation with LLM..."):
                interpretation = generate_interpretation(report)
                st.session_state['interpretation'] = interpretation
            st.success("Analysis and interpretation completed successfully!")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.error("Please try again with a different file or contact support if the issue persists.")

# Display Results
if st.session_state['analysis_report']:
    st.header("Analysis Results")
    
    # Create tabs for different analysis types
    tabs = st.tabs([tab for tab in analysis_types])
    
    for i, tab in enumerate(tabs):
        with tab:
            if analysis_types[i] == "Emotional Response":
                emotional_data = st.session_state['analysis_report']['emotional_response']
                
                # Plot emotional arc
                emotional_arc = pd.DataFrame(emotional_data['emotional_arc'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=emotional_arc['time'], y=emotional_arc['energy'],
                                       name='Energy', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=emotional_arc['time'], y=emotional_arc['pitch'],
                                       name='Pitch', line=dict(color='red')))
                fig.update_layout(
                    title='Emotional Arc Over Time',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Intensity',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_types[i] == "Engagement Patterns":
                engagement_data = st.session_state['analysis_report']['engagement_patterns']
                
                # Plot attention triggers
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=engagement_data['attention_triggers'],
                                       y=[1]*len(engagement_data['attention_triggers']),
                                       mode='markers', name='Attention Triggers'))
                fig.update_layout(
                    title='Attention Triggers Over Time',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Trigger Points',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display engagement metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tempo", f"{float(engagement_data['tempo']):.1f} BPM")
                with col2:
                    st.metric("Average Pause Duration", 
                             f"{float(np.mean(engagement_data['pause_durations'])):.2f} seconds")
                
            elif analysis_types[i] == "Brand Identity":
                brand_data = st.session_state['analysis_report']['brand_identity']
                
                # Display brand metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Voice Pitch Mean", f"{float(brand_data['voice_profile']['pitch_mean']):.2f}")
                with col2:
                    st.metric("Voice Energy", f"{float(brand_data['voice_profile']['energy_mean']):.2f}")
                with col3:
                    st.metric("Consistency Score", f"{float(brand_data['consistency_score']):.2f}")
                
            elif analysis_types[i] == "Psychological Impact":
                psych_data = st.session_state['analysis_report']['psychological_impact']
                
                # Display psychological metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Speech Rate", f"{float(psych_data['speech_rate']):.2f} segments/sec")
                with col2:
                    st.metric("Repetition Score", f"{float(psych_data['repetition_score']):.2f}")
                with col3:
                    st.metric("Emotional Intensity", f"{float(psych_data['emotional_intensity']):.2f}")
                
            elif analysis_types[i] == "Audience Reception":
                audience_data = st.session_state['analysis_report']['audience_reception']
                
                # Display audience metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tempo", f"{float(audience_data['tempo']):.1f} BPM")
                    st.metric("Clarity Score", f"{float(audience_data['clarity_score']):.2f}")
                with col2:
                    st.metric("Spectral Centroid", 
                             f"{float(audience_data['spectral_characteristics']['centroid']):.2f}")
                    st.metric("Spectral Rolloff", 
                             f"{float(audience_data['spectral_characteristics']['rolloff']):.2f}")

    # Show interpretation if available
    if 'interpretation' in st.session_state and st.session_state['interpretation']:
        st.subheader("LLM Evaluation & Suggestions")
        st.info(st.session_state['interpretation'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Audio Analysis Dashboard | Created with Streamlit</p>
    <p>For support or feedback, please contact the development team</p>
</div>
""", unsafe_allow_html=True) 