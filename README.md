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
- [Data Preprocessing](#data-preprocessing)  
- [Model Training](#model-training)  
- [Streamlit Web App](#streamlit-web-app)  
- [Deployment](#deployment)  
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


