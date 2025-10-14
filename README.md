# Review Sentiment Analyzer

<em>A web application that predicts the sentiment (positive or negative) of movie reviews using **Word2Vec embeddings** and **Logistic Regression**. Built with **Streamlit** for an interactive web interface.</em>
 
This project demonstrates **Natural Language Processing (NLP)**, machine learning, and model deployment skills.

---

## Table of Contents

- [Introduction](#introduction)  
- [Features](#features)  
- [Technology Stack](#technology-stack)  
- [Dataset](#dataset)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Data Preprocessing](#Preprocessing-&-Feature-Extraction (Optimized with Swifter))  
- [Streamlit Web App](#streamlit-web-app)  
- [Screenshots](#screenshots)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Introduction

This project is a **Movie Review Sentiment Analyzer** that classifies reviews as either **positive** or **negative**.  

The workflow is:

1. Preprocess movie reviews (tokenization, stopwords removal, lemmatization).  
2. Convert words to vectors using **Word2Vec embeddings**.  
3. Train a **Logistic Regression** model on these embeddings.  
4. Deploy the model in a **Streamlit** web app for live sentiment prediction.

---

## Features

- Predict sentiment for **custom movie reviews**  
- Interactive **web interface** with **Streamlit**  
- Pre-trained Word2Vec embeddings capture semantic meaning  
- Clean and user-friendly UI  
- Supports batch prediction (optional)  
- <em>**Optimized preprocessing using Swifter** for large datasets</em>

---

## Technology Stack

- **Python 3.10+**  
- **Libraries:** `numpy`, `pandas`, `nltk`, `gensim`, `scikit-learn`, `streamlit`, `swifter`  
- **Machine Learning:** Logistic Regression  
- **NLP:** Word2Vec embeddings  
- **Deployment:** Streamlit Cloud  

---

## Dataset

<em>The model can be trained on datasets such as:</em>

[![Kaggle Datasets](https://img.shields.io/badge/Kaggle-Movie_Reviews-lightgrey?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets)

[![IMDB Dataset](https://img.shields.io/badge/IMDB-Large_Movie_Review-blue?style=for-the-badge&logo=imdb)](https://ai.stanford.edu/~amaas/data/sentiment/)

---

**Columns required:**  

| Column Name | Description             |
|-------------|-------------------------|
| `review`    | Text of the movie review |
| `sentiment` | Label (`positive`/`negative`) |

**Note:** Make sure to preprocess the data (tokenization, stopwords removal, lemmatization) before training.

---

## Project Structure

```text
movie-review-sentiment-analyzer/
│
├── data/                     # Dataset files
│   └── movie_reviews.csv
├── models/                   # Saved models
│   ├── word2vec.model
│   └── logistic_model.pkl
├── app.py                     # Streamlit web app
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── screenshots/               # App screenshots
```
---
## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/movie-review-sentiment-analyzer.git
cd movie-review-sentiment-analyzer
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK resources**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## Preprocessing & Feature Extraction (Optimized with Swifter)

<em>This section covers the steps to clean movie reviews, tokenize them safely using the Word2Vec vocabulary, and convert them into document vectors for modeling.</em>

---

### 1️⃣ Clean and Preprocess Text

Steps applied to each review:

1. Lowercase all text  
2. Remove punctuation and special characters  
3. Tokenize sentences into words  
4. Remove stopwords  
5. Lemmatize words  

```python
import swifter
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation & special chars
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
    return tokens

# Apply preprocessing with Swifter
movie_df['tokens'] = movie_df['review'].swifter.apply(preprocess_text)
```
<em>Tip: Swifter automatically detects if parallel processing is possible, significantly speeding up operations on large datasets.</em>

### 2️⃣ Tokenize Reviews Using Word2Vec Vocabulary

After training the Word2Vec model, filter tokens to keep only words present in the vocabulary.
```python
# Extract vocabulary set from the Word2Vec model
vocab_set = set(model.wv.index_to_key)

# Safe tokenization function
def tokenize_review_safe(review, vocab):
    if not isinstance(review, str):
        return []
    return [w for w in review.split() if w in vocab]

# Apply safe tokenization with Swifter
movie_df['tokens'] = movie_df['review'].swifter.apply(
    lambda x: tokenize_review_safe(x, vocab_set)
)
```
Notes:
- Only words present in the Word2Vec vocabulary are kept.
- Handles non-string entries gracefully by returning an empty list.

### 3️⃣ Convert Tokenized Reviews to Document Vectors

Each review is represented as a fixed-size vector by averaging the embeddings of its tokens.
```python
import numpy as np

# Function to convert tokenized review into a document vector
def document_vector_fast(tokens):
    if len(tokens) == 0:
        return np.zeros(model.vector_size, dtype=np.float32)  # Handle empty token lists
    return np.mean(model.wv[tokens], axis=0)  # Average Word2Vec embeddings

# Apply the function using Swifter for faster computation
movie_vectors = np.vstack(
    movie_df['tokens'].swifter.apply(document_vector_fast).values
)
```
Notes:
- Returns a zero vector for empty token lists to avoid errors.
- Converts each review into a numerical vector suitable for machine learning models.
- Swifter significantly speeds up vectorization for large datasets.


