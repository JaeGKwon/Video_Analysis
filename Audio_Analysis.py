import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
from pydub import AudioSegment
import os
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import pandas as pd
from collections import Counter

class AdvancedAudioAnalyzer:
    def __init__(self, audio_path):
        """
        Initialize the AdvancedAudioAnalyzer with the path to the audio file
        """
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def emotional_response_analysis(self):
        """
        Analyze emotional response through various audio features
        """
        # 1. Tone Analysis
        # Extract pitch and energy features
        pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr)
        energy = librosa.feature.rms(y=self.y)[0]
        
        # Calculate emotional indicators
        pitch_mean = np.mean(pitches, axis=1)
        energy_mean = np.mean(energy)
        
        # 2. Emotional Arc Mapping
        # Create time windows for emotional progression
        window_size = int(self.sr * 1.0)  # 1-second windows
        emotional_arc = []
        
        for i in range(0, len(self.y), window_size):
            window = self.y[i:i+window_size]
            if len(window) == window_size:
                # Calculate features for this window
                window_energy = librosa.feature.rms(y=window)[0]
                window_pitch = librosa.piptrack(y=window, sr=self.sr)[0]
                emotional_arc.append({
                    'time': i/self.sr,
                    'energy': np.mean(window_energy),
                    'pitch': np.mean(window_pitch)
                })
        
        return {
            'pitch_profile': pitch_mean,
            'energy_profile': energy_mean,
            'emotional_arc': emotional_arc
        }
    
    def engagement_pattern_analysis(self):
        """
        Analyze engagement patterns in the audio
        """
        # 1. Attention Triggers
        # Detect significant volume changes
        rms = librosa.feature.rms(y=self.y)[0]
        peaks, _ = find_peaks(rms, height=np.mean(rms) + np.std(rms))
        
        # 2. Hook Effectiveness
        # Analyze repetition patterns
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        # 3. Narrative Clarity
        # Analyze speech rate and pauses
        intervals = librosa.effects.split(self.y, top_db=20)
        pause_durations = [(end - start)/self.sr for start, end in intervals]
        
        return {
            'attention_triggers': peaks/self.sr,
            'tempo': tempo,
            'beats': beats/self.sr,
            'pause_durations': pause_durations
        }
    
    def brand_identity_analysis(self, reference_audio_path=None):
        """
        Analyze brand identity through audio characteristics
        """
        # 1. Sonic Branding Recognition
        # Extract MFCCs for brand signature analysis
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        
        # 2. Voice Brand Fit
        # Analyze voice characteristics
        pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr)
        voice_profile = {
            'pitch_mean': np.mean(pitches),
            'pitch_std': np.std(pitches),
            'energy_mean': np.mean(librosa.feature.rms(y=self.y)[0])
        }
        
        # 3. Audio Consistency
        consistency_score = 0
        if reference_audio_path:
            ref_y, ref_sr = librosa.load(reference_audio_path)
            ref_mfccs = librosa.feature.mfcc(y=ref_y, sr=ref_sr, n_mfcc=13)
            # Calculate similarity between current and reference audio
            consistency_score = np.mean(np.abs(mfccs - ref_mfccs))
        
        return {
            'mfcc_profile': mfccs,
            'voice_profile': voice_profile,
            'consistency_score': consistency_score
        }
    
    def psychological_impact_analysis(self):
        """
        Analyze psychological impact of the audio
        """
        # 1. Cognitive Load
        # Analyze speech rate and information density
        intervals = librosa.effects.split(self.y, top_db=20)
        speech_segments = [(end - start)/self.sr for start, end in intervals]
        speech_rate = len(speech_segments) / (len(self.y)/self.sr)
        
        # 2. Memorability Factors
        # Analyze repetition patterns
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        repetition_score = np.std(onset_env)  # Higher variation might indicate more memorable patterns
        
        # 3. Persuasion Triggers
        # Analyze emotional intensity variations
        rms = librosa.feature.rms(y=self.y)[0]
        emotional_intensity = np.std(rms)
        
        return {
            'speech_rate': speech_rate,
            'repetition_score': repetition_score,
            'emotional_intensity': emotional_intensity
        }
    
    def audience_reception_analysis(self, demographic_data=None):
        """
        Analyze potential audience reception
        """
        # 1. Demographic Response Patterns
        # Analyze audio characteristics that might appeal to different demographics
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        pitch_mean = np.mean(librosa.piptrack(y=self.y, sr=self.sr)[0])
        
        # 2. Cultural Context Fit
        # Analyze musical elements that might have cultural significance
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        
        # 3. Accessibility Experience
        # Analyze clarity and comprehensibility
        clarity_score = np.mean(librosa.feature.spectral_contrast(y=self.y, sr=self.sr))
        
        return {
            'tempo': tempo,
            'pitch_profile': pitch_mean,
            'spectral_characteristics': {
                'centroid': np.mean(spectral_centroid),
                'rolloff': np.mean(spectral_rolloff)
            },
            'clarity_score': clarity_score
        }
    
    def generate_comprehensive_report(self, reference_audio_path=None, demographic_data=None):
        """
        Generate a comprehensive analysis report
        """
        report = {
            'emotional_response': self.emotional_response_analysis(),
            'engagement_patterns': self.engagement_pattern_analysis(),
            'brand_identity': self.brand_identity_analysis(reference_audio_path),
            'psychological_impact': self.psychological_impact_analysis(),
            'audience_reception': self.audience_reception_analysis(demographic_data)
        }
        
        # Save report to CSV
        df = pd.DataFrame(report)
        df.to_csv('audio_analysis_report.csv', index=False)
        
        return report

# Example usage
if __name__ == "__main__":
    # Replace with your audio file path
    audio_path = "path_to_your_audio_file.mp3"
    reference_audio_path = "path_to_reference_audio.mp3"  # Optional
    
    analyzer = AdvancedAudioAnalyzer(audio_path)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(
        reference_audio_path=reference_audio_path,
        demographic_data=None  # Optional demographic data
    )
    
    # Print key findings
    print("\nKey Analysis Findings:")
    print(f"Emotional Intensity: {report['psychological_impact']['emotional_intensity']:.2f}")
    print(f"Speech Rate: {report['psychological_impact']['speech_rate']:.2f} segments/second")
    print(f"Tempo: {report['audience_reception']['tempo']:.2f} BPM")
    print(f"Clarity Score: {report['audience_reception']['clarity_score']:.2f}")
