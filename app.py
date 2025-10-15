import streamlit as st
import joblib
from gensim.models import Word2Vec
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# -------------------------
# NLTK Downloads
# -------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="💬", layout="centered")

# -------------------------
# Load Models
# -------------------------
model = joblib.load("final_model.pkl")
w2v_model = Word2Vec.load("word2vec.model")
tfidf = joblib.load("tfidf.pkl")

# -------------------------
# Preprocessing
# -------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_review(review):
    if not isinstance(review, str) or review.strip() == "":
        return []
    review = re.sub(r'<.*?>', ' ', review)
    review = re.sub(r'[^a-zA-Z\s]', '', review.lower())
    tokens = word_tokenize(review)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def weighted_vector(tokens, w2v_model, tfidf_vectorizer):
    word2weight = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    vectors, weights = [], []
    for word in tokens:
        if word in w2v_model.wv and word in word2weight:
            vectors.append(w2v_model.wv[word])
            weights.append(word2weight[word])
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.average(vectors, axis=0, weights=weights)

# -------------------------
# UI Colors & CSS
# -------------------------
background = "linear-gradient(135deg, #eef2ff, #f8fafc)"
text_color = "#000000"
box_bg = "#ffffff"
border_color = "#e5e7eb"
card_bg = "#f9fafb"
subtitle_color = "#6b7280"

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
# Header
# -------------------------
st.markdown("""
<div class="header">
    <h1 class="title">IMDB Sentiment Analyzer</h1>
    <p class="subtitle">Analyze movie reviews instantly using AI-powered sentiment analysis 🎬</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Main Box
# -------------------------
st.markdown("<div class='main-box'>", unsafe_allow_html=True)
st.markdown("<h4><em>Enter your review</em></h4>", unsafe_allow_html=True)

review_input = st.text_area(
    "Movie Review Input",
    height=120,
    placeholder="Example: An amazing movie with a great cast and fantastic storyline!",
    label_visibility="collapsed"
)

col1, col2 = st.columns(2)

# Analyze Button
with col1:
    if st.button("Analyze", key="analyze"):
        review_text = review_input.strip()
        if review_text == "":
            st.warning("Please enter a review.")
        else:
            try:
                tokens = preprocess_review(review_text)
                vec = weighted_vector(tokens, w2v_model, tfidf).reshape(1, -1)
                prediction = model.predict(vec)[0]

                # Confidence (if model supports)
                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(vec)[0][prediction]
                    confidence_text = f" ({confidence*100:.1f}% confidence)"
                else:
                    confidence_text = ""

                sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
                color = "#16a34a" if prediction == 1 else "#dc2626"

                st.markdown(
                    f"<div class='result-box' style='color:{color};'>Sentiment: {sentiment}{confidence_text}</div>",
                    unsafe_allow_html=True
                )

                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append((review_text, sentiment))
                st.session_state.history = st.session_state.history[-20:]  # last 20 reviews

            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")

# Clear History Button
with col2:
    if st.button("Clear History", key="clear"):
        st.session_state.history = []
        st.success("History cleared successfully!")

st.markdown("</div>", unsafe_allow_html=True)  # Close main-box

# -------------------------
# History Section
# -------------------------
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

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    Made with ❤️ using <strong>Streamlit</strong> & <strong>Machine Learning</strong><br>
    Developed by <strong>Faisal Khan</strong>
</div>
""", unsafe_allow_html=True)
