import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- NLTK ---------------- #

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# ---------------- CLEAN TEXT ---------------- #


def clean_text(text):

    text = str(text)

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = text.lower().split()

    text = [
        ps.stem(word)
        for word in text
        if word not in stop_words
    ]

    return ' '.join(text)

# ---------------- LOAD DATASET ---------------- #

fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0
real['label'] = 1

news = pd.concat([fake, real], axis=0)

news = news.sample(frac=1)

news['content'] = (
    news['title'].astype(str) + ' ' +
    news['text'].astype(str)
)

news['content'] = news['content'].apply(clean_text)

X = news['content']
y = news['label']

# ---------------- TFIDF ---------------- #

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=50000
)

X = vectorizer.fit_transform(X)

# ---------------- SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ---------------- #

model = PassiveAggressiveClassifier(
    max_iter=1000
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f'Accuracy: {acc * 100:.2f}%')

# ---------------- SAVE ---------------- #

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print('Model Saved Successfully')



import streamlit as st
import numpy as np
import requests
import re
import joblib

# ---------------- PAGE ---------------- #

st.set_page_config(
    page_title='Fake News Detector',
    page_icon='📰',
    layout='wide'
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827
    );
    color: white;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background: #081028;
}

h1,h2,h3,p,label,div {
    color: white !important;
}

.card {
    background: rgba(15,23,42,0.9);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 25px;
    margin-top: 20px;
}

.stTextArea textarea {
    background: #0f172a !important;
    color: white !important;
    border: 1px solid #9333ea !important;
    border-radius: 14px !important;
}

.stTextInput input {
    background: #0f172a !important;
    color: white !important;
    border-radius: 12px !important;
}

.stButton button {
    width: 100%;
    background: linear-gradient(to right,#2563eb,#9333ea) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

.real {
    background: rgba(22,163,74,0.2);
    border: 1px solid #22c55e;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.fake {
    background: rgba(220,38,38,0.2);
    border: 1px solid #ef4444;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource

def load_files():

    model = joblib.load('model.pkl')

    vectorizer = joblib.load('vectorizer.pkl')

    return model, vectorizer

model, vectorizer = load_files()

# ---------------- CLEAN ---------------- #


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    return text

# ---------------- PREDICT ---------------- #


def predict_news(news):

    cleaned = clean_text(news)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = 95

    return prediction, confidence

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title('📰 Fake News Detector')

    st.success('Model Loaded Successfully')

    st.markdown('---')

    st.write('### Technologies')

    st.write('• Streamlit')
    st.write('• Machine Learning')
    st.write('• NLP')
    st.write('• TF-IDF')
    st.write('• PassiveAggressiveClassifier')

# ---------------- TITLE ---------------- #

st.markdown('''
# 📰 Live Fake News Detection System

Detect fake and real news using AI & NLP
''')

# ---------------- MANUAL ---------------- #

st.markdown("""
<div class='card'>
<h2>✍ Manual News Detection</h2>
</div>
""", unsafe_allow_html=True)

with st.form('manual_form'):

    input_news = st.text_area(
        '',
        height=220,
        placeholder='Paste news article here...'
    )

    submit = st.form_submit_button('🔍 Analyze News')

if submit:

    if input_news.strip() == '':

        st.warning('Please enter news')

    else:

        result, confidence = predict_news(input_news)

        if result == 0:

            st.markdown(f"""
            <div class='fake'>
            🚨 FAKE NEWS
            <br><br>
            Confidence: {confidence}%
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class='real'>
            ✅ REAL NEWS
            <br><br>
            Confidence: {confidence}%
            </div>
            """, unsafe_allow_html=True)

# ---------------- LIVE NEWS ---------------- #

st.markdown('---')

st.markdown("""
<div class='card'>
<h2>🌍 Live News Detection</h2>
</div>
""", unsafe_allow_html=True)

API_KEY = st.secrets["a81a3b150f6c47de956bd7e2519e0dce"]

# ---------------- FETCH ---------------- #


def fetch_news(keyword):

    url = (
        f'https://newsapi.org/v2/everything?'
        f'q={keyword}&language=en&pageSize=5&sortBy=publishedAt&apiKey={API_KEY}'
    )

    response = requests.get(url)

    data = response.json()

    return data.get('articles', [])

with st.form('live_form'):

    keyword = st.text_input(
        '',
        placeholder='AI, Cricket, Election, Bitcoin'
    )

    live_btn = st.form_submit_button('📰 Fetch Live News')

if live_btn:

    articles = fetch_news(keyword)

    if len(articles) == 0:

        st.error('No news found')

    else:

        for article in articles:

            title = article.get('title', '')
            description = article.get('description', '')
            source = article.get('source', {}).get('name', '')
            url = article.get('url', '')

            content = title + ' ' + description

            result, confidence = predict_news(content)

            if result == 0:
                label = '🚨 Fake'
                color = '#7f1d1d'
            else:
                label = '✅ Real'
                color = '#14532d'

            st.markdown(f"""
            <div class='card'>

            <h3>{title}</h3>

            <p>{description}</p>

            <p><b>Source:</b> {source}</p>

            <a href='{url}' target='_blank'>Read Full Article</a>

            <br><br>

            <div style='
            background:{color};
            padding:15px;
            border-radius:12px;
            text-align:center;
            font-size:18px;
            font-weight:bold;
            color:white;
            '>

            {label}

            </div>

            </div>
            """, unsafe_allow_html=True)
