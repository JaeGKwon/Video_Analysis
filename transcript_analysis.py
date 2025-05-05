import streamlit as st
from transformers import pipeline
import openai

# Set your OpenAI API key (use st.secrets in production)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Sentiment and emotion pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

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

st.title("Ad Transcript Content Analyzer")

st.write("""
Paste your ad transcript below or upload a .txt file.\n
All analyses will be run on the provided transcript.
""")

uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])
if uploaded_file is not None:
    transcript = uploaded_file.read().decode("utf-8")
else:
    transcript = st.text_area("Paste transcript here:", height=300)

if st.button("Analyze Transcript"):
    if not transcript.strip():
        st.error("Please provide a transcript by pasting or uploading a file.")
    else:
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
