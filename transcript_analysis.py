import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import openai

# Set your OpenAI API key (use st.secrets in production)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Sentiment and emotion pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

def get_transcript_from_url(url):
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error extracting transcript: {e}"

def analyze_sentiment(text):
    return sentiment_analyzer(text[:512])  # Truncate for model input limit

def analyze_emotion(text):
    return emotion_analyzer(text[:512])  # Truncate for model input limit

def llm_analysis(text, analysis_type):
    prompt = (
        f"Analyze the following ad transcript for {analysis_type}. "
        "Give a concise summary and actionable feedback for a marketing team:\n"
        f"{text}"
    )
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content

st.title("YouTube Ad Content Analyzer")

youtube_url = st.text_input("Enter YouTube Video URL:")

if st.button("Analyze Ad"):
    with st.spinner("Extracting transcript..."):
        transcript = get_transcript_from_url(youtube_url)
        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            st.success("Transcript extracted!")
            st.subheader("Transcript")
            st.write(transcript)

            with st.spinner("Running analyses..."):
                sentiment = analyze_sentiment(transcript)
                emotion = analyze_emotion(transcript)
                key_message = llm_analysis(transcript, "key message extraction")
                persuasion = llm_analysis(transcript, "persuasion and rhetorical devices")
                brand = llm_analysis(transcript, "brand consistency")
                compliance = llm_analysis(transcript, "compliance and safety")
                inclusion = llm_analysis(transcript, "diversity and inclusion")
                cta = llm_analysis(transcript, "call-to-action effectiveness")

            st.header("Analysis Results")
            st.subheader("Sentiment Analysis")
            st.json(sentiment)
            st.subheader("Emotion Detection")
            st.json(emotion)
            st.subheader("Key Message Extraction")
            st.info(key_message)
            st.subheader("Persuasion & Rhetorical Devices")
            st.info(persuasion)
            st.subheader("Brand Consistency")
            st.info(brand)
            st.subheader("Compliance & Safety")
            st.info(compliance)
            st.subheader("Diversity & Inclusion")
            st.info(inclusion)
            st.subheader("Call-to-Action Effectiveness")
            st.info(cta)
