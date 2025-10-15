import streamlit as st
import joblib
from gensim.models import Word2Vec
import numpy as np

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="💬", layout="centered")

# -------------------------
# Load Models
# -------------------------
model = joblib.load("final_model.pkl")
w2v_model = Word2Vec.load("word2vec.model")

def vectorize_review(review, w2v_model):
    tokens = review.lower().split()
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vectors, axis=0)

# -------------------------
# Fixed Theme Colors (Light)
# -------------------------
background = "linear-gradient(135deg, #eef2ff, #f8fafc)"
text_color = "#000000"
box_bg = "#ffffff"
border_color = "#e5e7eb"
card_bg = "#f9fafb"
subtitle_color = "#6b7280"
analyze_btn = "linear-gradient(90deg, #2563eb, #1e40af)"
clear_btn = "linear-gradient(90deg, #dc2626, #b91c1c)"
button_text_color = "#000000"


# -------------------------
# CSS with Fade-In & Styling
# -------------------------
st.markdown(f"""
<style>
.stApp {{
    background: {background};
    color: {text_color};
    font-family: 'Inter', sans-serif;
    transition: all 0.5s ease-in-out;
}}

.header, .main-box, .footer {{
    opacity: 0;
    animation: fadeIn 0.7s forwards;
}}
@keyframes fadeIn {{
    to {{ opacity: 1; }}
}}

.title {{
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(to right, #2563eb, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.subtitle {{
    font-size: 16px;
    color: {subtitle_color};
    margin-top: -8px;
}}

# .main-box {{
#     background-color: {box_bg};
#     border: 1px solid {border_color};
#     border-radius: 18px;
#     box-shadow: 0 6px 20px rgba(0,0,0,0.08);
#     padding: 35px;
#     width: 85%;
#     margin: auto;
#     transition: all 0.5s ease-in-out;
# }}

.stButton > button {{
    background: linear-gradient(90deg, #1d4ed8, #2563eb); /* Professional blue gradient */
    color: #ffffff !important; /* White text */
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6em 1.2em !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease-in-out !important;
    box-shadow: 0 3px 8px rgba(37, 99, 235, 0.3); /* Soft blue glow */
}}
.stButton > button:hover {{
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 6px 14px rgba(37, 99, 235, 0.4);
    background: linear-gradient(90deg, #2563eb, #1e40af); /* Slightly darker on hover */
}}
.stButton > button:active {{
    transform: scale(0.98);
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3);
}}

.result-box {{
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    margin-top: 30px;
    padding: 15px;
    border-radius: 10px;
    background-color: {card_bg};
    transition: all 0.5s ease-in-out;
}}

.history-section {{ margin-top: 25px; }}
.history-card {{
    background: {card_bg};
    border-radius: 8px;
    border: 1px solid {border_color};
    padding: 12px 14px;
    margin-top: 10px;
    transition: all 0.3s ease-in-out;
}}
.history-card:hover {{ transform: scale(1.02); }}

.footer {{
    text-align: center;
    padding: 18px 0;
    font-size: 14px;
    color: {subtitle_color};
}}
.footer strong {{
    background: linear-gradient(90deg, #2563eb, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header Section
# -------------------------
st.markdown("""
<div class="header">
    <h1 class="title">IMDB Sentiment Analyzer</h1>
    <p class="subtitle">Analyze movie reviews instantly using AI-powered sentiment analysis 🎬</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Main Content Box
# -------------------------
st.markdown("<div class='main-box'>", unsafe_allow_html=True)
st.markdown("<h4><em> Enter your review</em></h4>", unsafe_allow_html=True)

review_input = st.text_area(
    "Movie Review Input", 
    height=120, 
    placeholder="Example: An amazing movie with a great cast and fantastic storyline!",
    label_visibility="collapsed"
)

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button(" Analyze", key="analyze"):
        if review_input.strip() == "":
            st.warning(" Please enter a review.")
        else:
            vec = vectorize_review(review_input, w2v_model).reshape(1, -1)
            prediction = model.predict(vec)[0]
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
            color = "#16a34a" if prediction == 1 else "#dc2626"

            st.markdown(
                f"<div class='result-box' style='color:{color};'>Sentiment: {sentiment}</div>",
                unsafe_allow_html=True
            )

            if "history" not in st.session_state:
                st.session_state.history = []
            # Store both review text and sentiment
            st.session_state.history.append((review_input, sentiment))

with col2:
    if st.button(" Clear History", key="clear"):
        st.session_state.history = []
        st.success("History cleared successfully!")

st.markdown("</div>", unsafe_allow_html=True)  # close main-box

# History Section
if "history" in st.session_state and len(st.session_state.history) > 0:
    st.markdown("<div class='history-section'>", unsafe_allow_html=True)
    st.markdown("### Prediction History")

    for i, (review_text, sentiment) in enumerate(reversed(st.session_state.history), 1):
        color = "#10b981" if "Positive" in sentiment else "#ef4444"
        st.markdown(f"""
        <div class='history-card'>
            <strong style='color:{color}'>{i}. {sentiment}</strong><br>
            <span>{review_text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    Made with ❤️ using <strong>Streamlit</strong> & <strong>Machine Learning</strong><br>
    Developed by <strong>Faisal Khan</strong>
</div>
""", unsafe_allow_html=True)