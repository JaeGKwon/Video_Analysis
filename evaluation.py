import json
import streamlit as st

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
    # Define the dimensions to look for
    dimensions = [
        "Tone & Messaging",
        "Emotional Narrative",
        "Visual Execution",
        "CTA Effectiveness",
        "Target Market Fit",
        "Competitive Differentiation"
    ]
    # Try to map LLM Q&A keys to these dimensions
    # (You may want to customize this mapping based on your LLM output)
    mapping = {
        "Tone & Messaging": ["tone", "messaging", "Tone", "Messaging"],
        "Emotional Narrative": ["emotional", "narrative", "storytelling"],
        "Visual Execution": ["visual", "execution", "direction", "realism"],
        "CTA Effectiveness": ["cta", "call to action", "effectiveness"],
        "Target Market Fit": ["target", "market", "fit", "homeowners", "millennials"],
        "Competitive Differentiation": ["competitive", "differentiation", "price-first", "brands"]
    }
    # Aggregate answers from all screenshots (could average or take the best)
    summary = {dim: {"score": 0, "note": ""} for dim in dimensions}
    for entry in final_output:
        llm_qa = entry.get("llm_qa", {})
        if isinstance(llm_qa, dict):
            for dim in dimensions:
                for key in llm_qa:
                    if any(k.lower() in key.lower() for k in mapping[dim]):
                        score, note = extract_score_and_note(llm_qa[key])
                        # Take the highest score/note found for each dimension
                        if score > summary[dim]["score"]:
                            summary[dim]["score"] = score
                            summary[dim]["note"] = note
    # Render the scorecard as a markdown table
    md = "| Dimension | Score | Notes |\n|---|---|---|\n"
    for dim in dimensions:
        stars = STAR.get(summary[dim]["score"], STAR[0])
        note = summary[dim]["note"]
        md += f"| {dim} | {stars} | {note} |\n"
    st.markdown("**Final Evaluation Scorecard**")
    st.markdown(md)

# Streamlit UI
st.title("Evaluation Scorecard Generator")

uploaded = st.file_uploader("Upload final_output.json", type=["json"])
if uploaded:
    data = json.load(uploaded)
    render_scorecard(data)
else:
    st.info("Please upload a final_output.json file generated from Main.py.") 