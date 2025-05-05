import streamlit as st
import os
from Audio_Analysis import AdvancedAudioAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Audio Analysis Dashboard", layout="wide")
st.title("🎵 Audio Analysis Dashboard")

# Initialize session state
if 'audio_file' not in st.session_state:
    st.session_state['audio_file'] = None
if 'analyzer' not in st.session_state:
    st.session_state['analyzer'] = None
if 'analysis_report' not in st.session_state:
    st.session_state['analysis_report'] = None

# File uploader
st.header("Upload Audio")
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg"])

if audio_file is not None:
    # Save the uploaded file temporarily
    temp_path = f"temp_{audio_file.name}"
    with open(temp_path, "wb") as f:
        f.write(audio_file.read())
    
    st.session_state['audio_file'] = temp_path
    st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")

# Analysis Options
st.header("Analysis Options")
analysis_types = st.multiselect(
    "Select analysis types:",
    ["Emotional Response", "Engagement Patterns", "Brand Identity", 
     "Psychological Impact", "Audience Reception"],
    default=["Emotional Response", "Engagement Patterns"]
)

# Reference audio for brand identity analysis
if "Brand Identity" in analysis_types:
    st.subheader("Brand Identity Analysis")
    reference_audio = st.file_uploader("Upload reference audio for brand comparison", 
                                     type=["mp3", "wav", "m4a", "ogg"])
    if reference_audio:
        ref_path = f"temp_ref_{reference_audio.name}"
        with open(ref_path, "wb") as f:
            f.write(reference_audio.read())
    else:
        ref_path = None

# Demographic data for audience reception analysis
if "Audience Reception" in analysis_types:
    st.subheader("Demographic Data")
    demographic_data = {
        "age_group": st.selectbox("Target Age Group", 
                                ["18-24", "25-34", "35-44", "45-54", "55+"]),
        "gender": st.multiselect("Target Gender", ["Male", "Female", "Other"]),
        "region": st.text_input("Target Region", "Global")
    }
else:
    demographic_data = None

# Analysis Button
if st.button("Run Analysis") and st.session_state['audio_file']:
    with st.spinner("Analyzing audio..."):
        try:
            # Initialize analyzer
            analyzer = AdvancedAudioAnalyzer(st.session_state['audio_file'])
            st.session_state['analyzer'] = analyzer
            
            # Run selected analyses
            report = {}
            
            if "Emotional Response" in analysis_types:
                report['emotional_response'] = analyzer.emotional_response_analysis()
                
            if "Engagement Patterns" in analysis_types:
                report['engagement_patterns'] = analyzer.engagement_pattern_analysis()
                
            if "Brand Identity" in analysis_types:
                report['brand_identity'] = analyzer.brand_identity_analysis(ref_path)
                
            if "Psychological Impact" in analysis_types:
                report['psychological_impact'] = analyzer.psychological_impact_analysis()
                
            if "Audience Reception" in analysis_types:
                report['audience_reception'] = analyzer.audience_reception_analysis(demographic_data)
            
            st.session_state['analysis_report'] = report
            st.success("Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

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
                fig.update_layout(title='Emotional Arc Over Time',
                                xaxis_title='Time (seconds)',
                                yaxis_title='Intensity')
                st.plotly_chart(fig)
                
            elif analysis_types[i] == "Engagement Patterns":
                engagement_data = st.session_state['analysis_report']['engagement_patterns']
                
                # Plot attention triggers
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=engagement_data['attention_triggers'],
                                       y=[1]*len(engagement_data['attention_triggers']),
                                       mode='markers', name='Attention Triggers'))
                fig.update_layout(title='Attention Triggers Over Time',
                                xaxis_title='Time (seconds)',
                                yaxis_title='Trigger Points')
                st.plotly_chart(fig)
                
                # Display engagement metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tempo", f"{engagement_data['tempo']:.1f} BPM")
                with col2:
                    st.metric("Average Pause Duration", 
                             f"{np.mean(engagement_data['pause_durations']):.2f} seconds")
                
            elif analysis_types[i] == "Brand Identity":
                brand_data = st.session_state['analysis_report']['brand_identity']
                
                # Display brand metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Voice Pitch Mean", f"{brand_data['voice_profile']['pitch_mean']:.2f}")
                with col2:
                    st.metric("Voice Energy", f"{brand_data['voice_profile']['energy_mean']:.2f}")
                with col3:
                    st.metric("Consistency Score", f"{brand_data['consistency_score']:.2f}")
                
            elif analysis_types[i] == "Psychological Impact":
                psych_data = st.session_state['analysis_report']['psychological_impact']
                
                # Display psychological metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Speech Rate", f"{psych_data['speech_rate']:.2f} segments/sec")
                with col2:
                    st.metric("Repetition Score", f"{psych_data['repetition_score']:.2f}")
                with col3:
                    st.metric("Emotional Intensity", f"{psych_data['emotional_intensity']:.2f}")
                
            elif analysis_types[i] == "Audience Reception":
                audience_data = st.session_state['analysis_report']['audience_reception']
                
                # Display audience metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tempo", f"{audience_data['tempo']:.1f} BPM")
                    st.metric("Clarity Score", f"{audience_data['clarity_score']:.2f}")
                with col2:
                    st.metric("Spectral Centroid", 
                             f"{audience_data['spectral_characteristics']['centroid']:.2f}")
                    st.metric("Spectral Rolloff", 
                             f"{audience_data['spectral_characteristics']['rolloff']:.2f}")

# Cleanup
if st.session_state['audio_file'] and os.path.exists(st.session_state['audio_file']):
    os.remove(st.session_state['audio_file'])
if 'ref_path' in locals() and os.path.exists(ref_path):
    os.remove(ref_path) 