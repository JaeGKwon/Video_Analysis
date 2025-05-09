import json
import streamlit as st
import openai
import base64
import os

# Mapping for score to stars
STAR = {
    1: '★☆☆☆☆',
    2: '★★☆☆☆',
    3: '★★★☆☆',
    4: '★★★★☆',
    5: '★★★★★',
    0: '☆☆☆☆☆'
}

def parse_score(text):
    """
    Extract a numeric score (1-5) from a string, fallback to 0 if not found.
    """
    import re
    match = re.search(r'(\d+)[/|\\-]?(10|5)?', text)
    if match:
        score = int(match.group(1))
        if match.group(2) == '10':
            return round(score / 2)  # Convert 10-point scale to 5
        return min(score, 5)
    return 0

def extract_score_and_note(answer):
    """
    Try to extract a score and a note from an LLM answer string.
    """
    if isinstance(answer, dict):
        # If answer is a dict, try to get 'score' and 'note' keys
        score = answer.get('score', 0)
        note = answer.get('note', str(answer))
        return score, note
    if not isinstance(answer, str):
        return 0, str(answer)
    # Try to split score and note
    parts = answer.split(':', 1)
    score = parse_score(parts[0])
    note = parts[1].strip() if len(parts) > 1 else answer
    return score, note

def render_scorecard(final_output):
    dimensions = [
        "Tone & Messaging",
        "Emotional Narrative",
        "Visual Execution",
        "CTA Effectiveness",
        "Target Market Fit",
        "Competitive Differentiation"
    ]
    mapping = {
        "Tone & Messaging": ["tone", "messaging", "Tone", "Messaging"],
        "Emotional Narrative": ["emotional", "narrative", "storytelling"],
        "Visual Execution": ["visual", "execution", "direction", "realism"],
        "CTA Effectiveness": ["cta", "call to action", "effectiveness"],
        "Target Market Fit": ["target", "market", "fit", "homeowners", "millennials"],
        "Competitive Differentiation": ["competitive", "differentiation", "price-first", "brands"]
    }
    summary = {dim: {"score": 0, "note": ""} for dim in dimensions}
    for entry in final_output:
        # Use persona-driven answers if available
        llm_qa = entry.get("llm_qa_persona") or entry.get("llm_qa", {})
        if isinstance(llm_qa, dict):
            for dim in dimensions:
                for key in llm_qa:
                    if any(k.lower() in key.lower() for k in mapping[dim]):
                        score, note = extract_score_and_note(llm_qa[key])
                        if score > summary[dim]["score"]:
                            summary[dim]["score"] = score
                            summary[dim]["note"] = note
    md = "| Dimension | Score | Notes |\n|---|---|---|\n"
    for dim in dimensions:
        stars = STAR.get(summary[dim]["score"], STAR[0])
        note = summary[dim]["note"]
        md += f"| {dim} | {stars} | {note} |\n"
    st.markdown("**Final Evaluation Scorecard**")
    st.markdown(md)

# Example Experian Mosaic persona profiles (customize as needed)
MOSAIC_PROFILES = {
    "Power Elite": {
        "State": "CA",
        "Urban/Rural": "Urban",
        "Age": 52,
        "Gender": "Male",
        "Income": "> $200k",
        "Marital Status": "Married",
        "Number of Kids": 2,
        "Education": "Doctorate",
        "Ethnicity": "White"
    },
    "Flourishing Families": {
        "State": "TX",
        "Urban/Rural": "Suburban",
        "Age": 40,
        "Gender": "Female",
        "Income": "$100k-$200k",
        "Marital Status": "Married",
        "Number of Kids": 3,
        "Education": "Bachelor's",
        "Ethnicity": "White"
    },
    "Suburban Style": {
        "State": "IL",
        "Urban/Rural": "Suburban",
        "Age": 36,
        "Gender": "Female",
        "Income": "$50k-$100k",
        "Marital Status": "Married",
        "Number of Kids": 2,
        "Education": "Some College",
        "Ethnicity": "Hispanic"
    },
    "Career & Family": {
        "State": "NY",
        "Urban/Rural": "Urban",
        "Age": 34,
        "Gender": "Male",
        "Income": "$100k-$200k",
        "Marital Status": "Single",
        "Number of Kids": 0,
        "Education": "Master's",
        "Ethnicity": "Asian"
    },
    "Rural Heritage": {
        "State": "Other",
        "Urban/Rural": "Rural",
        "Age": 58,
        "Gender": "Male",
        "Income": "$25k-$50k",
        "Marital Status": "Married",
        "Number of Kids": 1,
        "Education": "High School",
        "Ethnicity": "White"
    },
    "Urban Ambition": {
        "State": "CA",
        "Urban/Rural": "Urban",
        "Age": 29,
        "Gender": "Non-binary",
        "Income": "$50k-$100k",
        "Marital Status": "Single",
        "Number of Kids": 0,
        "Education": "Bachelor's",
        "Ethnicity": "Black"
    }
}

# Streamlit UI
st.title("Evaluation Scorecard Generator")

uploaded = st.file_uploader("Upload final_output.json", type=["json"])

st.markdown("### Evaluator Profile")
# Experian Mosaic Persona selection first
mosaic_persona = st.selectbox("Experian Mosaic Persona", [
    "",
    "Power Elite",
    "Flourishing Families",
    "Suburban Style",
    "Career & Family",
    "Rural Heritage",
    "Urban Ambition",
    "Other"
])
st.markdown("---")
col1, col2 = st.columns(2)
# Set defaults based on persona
defaults = MOSAIC_PROFILES.get(mosaic_persona, {})
with col1:
    state = st.selectbox("Location (State)", ["", "CA", "NY", "TX", "FL", "IL", "Other"],
        index=["", "CA", "NY", "TX", "FL", "IL", "Other"].index(defaults.get("State", "")) if defaults.get("State") in ["CA", "NY", "TX", "FL", "IL", "Other"] else 0)
    urban_rural = st.selectbox("Urban/Rural", ["", "Urban", "Suburban", "Rural"],
        index=["", "Urban", "Suburban", "Rural"].index(defaults.get("Urban/Rural", "")) if defaults.get("Urban/Rural") in ["Urban", "Suburban", "Rural"] else 0)
    age = st.number_input("Age", min_value=0, max_value=120, value=defaults.get("Age", 35))
    gender = st.selectbox("Gender", ["", "Male", "Female", "Non-binary", "Other", "Prefer not to say"],
        index=["", "Male", "Female", "Non-binary", "Other", "Prefer not to say"].index(defaults.get("Gender", "")) if defaults.get("Gender") in ["Male", "Female", "Non-binary", "Other", "Prefer not to say"] else 0)
    marital_status = st.selectbox("Marital Status", ["", "Single", "Married", "Divorced", "Widowed", "Other"],
        index=["", "Single", "Married", "Divorced", "Widowed", "Other"].index(defaults.get("Marital Status", "")) if defaults.get("Marital Status") in ["Single", "Married", "Divorced", "Widowed", "Other"] else 0)
    num_kids = st.number_input("Number of Kids", min_value=0, max_value=10, value=defaults.get("Number of Kids", 0))
with col2:
    income = st.selectbox("Income", ["", "< $25k", "$25k-$50k", "$50k-$100k", "$100k-$200k", "> $200k"],
        index=["", "< $25k", "$25k-$50k", "$50k-$100k", "$100k-$200k", "> $200k"].index(defaults.get("Income", "")) if defaults.get("Income") in ["< $25k", "$25k-$50k", "$50k-$100k", "$100k-$200k", "> $200k"] else 0)
    education = st.selectbox("Education Level", ["", "High School", "Some College", "Bachelor's", "Master's", "Doctorate", "Other"],
        index=["", "High School", "Some College", "Bachelor's", "Master's", "Doctorate", "Other"].index(defaults.get("Education", "")) if defaults.get("Education") in ["High School", "Some College", "Bachelor's", "Master's", "Doctorate", "Other"] else 0)
    ethnicity = st.selectbox("Ethnicity", ["", "White", "Black", "Hispanic", "Asian", "Other", "Prefer not to say"],
        index=["", "White", "Black", "Hispanic", "Asian", "Other", "Prefer not to say"].index(defaults.get("Ethnicity", "")) if defaults.get("Ethnicity") in ["White", "Black", "Hispanic", "Asian", "Other", "Prefer not to say"] else 0)

profile = {
    "State": state,
    "Urban/Rural": urban_rural,
    "Age": age,
    "Gender": gender,
    "Income": income,
    "Marital Status": marital_status,
    "Number of Kids": num_kids,
    "Education": education,
    "Ethnicity": ethnicity,
    "Experian Mosaic Persona": mosaic_persona
}

evaluate = st.button("Evaluate")

# Set your OpenAI API key (assume it's in Streamlit secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_llm_qa_persona(image_path, questions_path, persona_profile):
    """
    Use GPT-4o to answer a set of questions about an image, from the perspective of the given persona.
    """
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    with open(questions_path, "r") as f:
        questions = f.read().strip()
    persona_context = (
        f"You are evaluating this as a persona with the following profile: {json.dumps(persona_profile)}. "
        "Answer the following questions from this perspective."
    )
    instruction = (
        "You are a chief creative officer. Evaluate the uploaded screenshot as if it is a frame from a streaming ad. "
        "For each of the following categories, provide a short, specific answer. If possible, include a numeric score from 1-10 (10 = best) for each category. "
        "Return your answers as a JSON object where each key is the category and each value is your answer."
    )
    prompt = persona_context + "\n\n" + instruction + "\n\n" + questions
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=1500
    )
    answer_text = response.choices[0].message.content
    try:
        start = answer_text.find('{')
        end = answer_text.rfind('}') + 1
        parsed = json.loads(answer_text[start:end])
        return parsed
    except Exception:
        return answer_text

if uploaded and evaluate:
    st.markdown("#### Selected Evaluator Profile:")
    st.json(profile)
    data = json.load(uploaded)
    st.info("Running persona-driven evaluation. This may take a while depending on the number of screenshots...")
    persona_results = []
    progress = st.progress(0)
    status = st.empty()
    logs = []
    for idx, entry in enumerate(data):
        img_path = entry["screenshot"]["file"]
        img_path_full = img_path if os.path.exists(img_path) else os.path.join("screenshots", img_path)
        questions_path = "questions.txt"
        status.text(f"Evaluating screenshot {idx+1} of {len(data)}...")
        logs.append(f"[START] Screenshot {idx+1}/{len(data)}: {img_path_full}")
        try:
            # Log the prompt
            with open(img_path_full, "rb") as img_file:
                img_bytes = img_file.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            with open(questions_path, "r") as f:
                questions = f.read().strip()
            persona_context = (
                f"You are evaluating this as a persona with the following profile: {json.dumps(profile)}. "
                "Answer the following questions from this perspective."
            )
            instruction = (
                "You are a chief creative officer. Evaluate the uploaded screenshot as if it is a frame from a streaming ad. "
                "For each of the following categories, provide a short, specific answer. If possible, include a numeric score from 1-10 (10 = best) for each category. "
                "Return your answers as a JSON object where each key is the category and each value is your answer."
            )
            prompt = persona_context + "\n\n" + instruction + "\n\n" + questions
            logs.append(f"[PROMPT] Screenshot {idx+1}: {prompt[:500]}... (truncated)")
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=1500
            )
            answer_text = response.choices[0].message.content
            try:
                start = answer_text.find('{')
                end = answer_text.rfind('}') + 1
                parsed = json.loads(answer_text[start:end])
                llm_qa_persona = parsed
            except Exception:
                llm_qa_persona = answer_text
            logs.append(f"[SUCCESS] Screenshot {idx+1}: LLM response received.")
        except Exception as e:
            llm_qa_persona = {"error": str(e)}
            logs.append(f"[ERROR] Screenshot {idx+1}: {str(e)}")
        persona_entry = dict(entry)
        persona_entry["llm_qa_persona"] = llm_qa_persona
        persona_results.append(persona_entry)
        progress.progress((idx+1)/len(data))
    status.success("Evaluation complete!")
    with st.expander("Show Evaluation Log"):
        for log in logs:
            st.write(log)
    render_scorecard(persona_results)
elif uploaded and not evaluate:
    st.info("Click 'Evaluate' to generate the scorecard.")
else:
    st.info("Please upload a final_output.json file generated from Main.py.") 