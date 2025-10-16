import streamlit as st
import joblib
from gensim.models import Word2Vec
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import sqlite3
from datetime import datetime

# -------------------------
# Setup
# -------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="IMDb Sentiment Analyzer", page_icon="🎬", layout="wide")

model = joblib.load("final_model.pkl")
w2v_model = Word2Vec.load("Word2Vec.model")
tfidf = joblib.load("tfidf.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

def weighted_vector(tokens, w2v_model, tfidf_vectorizer):
    word2weight = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    vectors, weights = [], []
    for w in tokens:
        if w in w2v_model.wv and w in word2weight:
            vectors.append(w2v_model.wv[w])
            weights.append(word2weight[w])
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    return np.average(vectors, axis=0, weights=weights)


# -------------------------
# SQLite Setup
# -------------------------
conn = sqlite3.connect("reviews.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT,
    sentiment TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
/* ------------------------- GENERAL ------------------------- */
html, body, .stApp {
    height: 100%;
    margin: 0;
    padding: 0;
    scroll-behavior: smooth;
    overflow-x: hidden;
    font-family: 'Inter', sans-serif;
}
/* ------------------------- HEADER SECTION ------------------------- */
.header-section {
    background: linear-gradient(120deg, #1e3a8a, #3b82f6);
    color: white;
    text-align: center;
    padding: 100px 20px;
    border-radius: 40px 40px 0px 0px;
}
.header-section h1 {
    font-size: 3rem;
    margin-bottom: 0.3rem;
}
.header-section p {
    font-size: 1.2rem;
    opacity: 0.9;
}

/* ------------------------- BODY SECTION ------------------------- */
# .body-section {
#     background: linear-gradient(135deg, #cceeff, #e0f7ff);
#     padding: 80px 40px;
#     margin: 0;       /* remove extra margin */
#     min-height: 50px;
# }

/* ------------------------- BUTTONS ------------------------- */
.stButton>button {
    background: linear-gradient(to right, #2196f3, #21cbf3);
    color: #f5f5f5;
    font-weight: 600;
    font-family: 'Roboto', sans-serif;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
}
.stButton>button:hover {
    background: linear-gradient(to right, #1976d2, #1e88e5);
    color: #ffffff;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}
/* ------------------------- TEXT AREA ------------------------- */
.review-section {
    background: #ffffff; /* white card background */
    padding: 20px 15px;       /* reduced padding */
    border-radius: 15px;     
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1); 
    margin-bottom: 10px;
    max-width: 900px;         /* limit max width */
    margin-left: auto;
    margin-right: auto;       /* center the box */
}
.review-section h2 {
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 15px;     /* smaller gap */
    font-size: 1.5rem;       /* smaller heading */
}
.review-section textarea {
    border-radius: 10px !important;
    padding: 10px !important; /* smaller padding inside textarea */
    font-size: 0.95rem !important;
    width: 100% !important;   /* full width of the card */
    height: 100px !important; /* smaller height */
}
     
/* ------------------------- FOOTER SECTION ------------------------- */
.footer-section {
    background-color: #1e293b;
    color: #cbd5e1;
    text-align: center;
    padding: 40px 20px;
    border-radius: 0 0 40px 40px;
    margin: 0; /* remove extra margin */
}

/* Optional: style links in footer */
.footer-section a {
    color: #60a5fa;
    text-decoration: none;
}
.footer-section a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)


# 🔹 HEADER SECTION
# =====================================================
st.markdown("""
<div class="header-section" id="header">
    <h1> IMDb Sentiment Analyzer</h1>
    <p>Discover what people think about movies — instantly analyze IMDb-style reviews using AI-powered sentiment prediction.</p>
    <a href="#body"><button style="background:#ffffff;color:#1e3a8a;padding:10px 20px;border:none;border-radius:10px;font-weight:600;cursor:pointer;">↓ Scroll to Analyzer</button></a>
</div>
""", unsafe_allow_html=True)


# BODY SECTION
st.markdown("<div class='body-section' id='body'>", unsafe_allow_html=True)

st.markdown("""
<div class="review-section">
    <h2>Enter Your Movie Review</h2>
</div>
""", unsafe_allow_html=True)

review_input = st.text_area(
    "",
    height=120,
    placeholder="Example: A brilliant movie with powerful performances!"
)

col1, col2 = st.columns(2)

# ANALYZE BUTTON
with col1:
    if st.button("Analyze"):
        if review_input.strip() == "":
            st.warning("Please enter a review.")
        else:
            try:
                tokens = preprocess_text(review_input)
                vec = weighted_vector(tokens, w2v_model, tfidf).reshape(1, -1)
                pred = model.predict(vec)[0]
                sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
                color = "#16a34a" if pred == 1 else "#dc2626"

                st.markdown(f"<h3 style='color:{color};text-align:center;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)

                # Save to database
                c.execute("INSERT INTO reviews (review_text, sentiment) VALUES (?, ?)", (review_input, sentiment))
                conn.commit()

            except Exception as e:
                st.error(f"Error: {e}")

# CLEAR HISTORY BUTTON
with col2:
    if st.button("Clear History"):
        c.execute("DELETE FROM reviews")
        conn.commit()
        st.info("History cleared!")

# SHOW HISTORY
st.markdown("### 🕒 Previous Analysis")
c.execute("SELECT review_text, sentiment, timestamp FROM reviews ORDER BY id DESC LIMIT 20")
rows = c.fetchall()
if rows:
    for i, (txt, sent, ts) in enumerate(rows, 1):
        color = "#10b981" if "Positive" in sent else "#ef4444"
        st.markdown(f"<div style='border-left:5px solid {color};padding:10px;margin:5px 0;background:#f1f5f9;border-radius:6px;'><b style='color:{color}'>{sent}</b> <span style='font-size:12px;color:gray;'>({ts})</span><br>{txt}</div>", unsafe_allow_html=True)
else:
    st.info("No reviews yet.")

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER SECTION
# =====================================================
st.markdown("""
<div class="footer-section" id="footer">
    <p>Made with ❤️ using <b>Streamlit</b> & <b>Machine Learning</b></p>
    <p>Developed by <b>Faisal Khan</b> | <a href="https://github.com/yourusername" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)