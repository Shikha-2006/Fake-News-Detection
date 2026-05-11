import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import time

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #3b82f6;
    font-size: 16px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #2563eb, #7c3aed);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

 
nltk.download('stopwords')


ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()

    content = [
        ps.stem(word)
        for word in content
        if word not in stop_words
    ]

    return ' '.join(content)


@st.cache_resource
def train_model():

    news_df = pd.read_csv("train.csv")
    news_df = news_df.fillna('')

    # Combine author + title
    news_df['content'] = (
        news_df['author'] + ' ' + news_df['title']
    )

    # Apply preprocessing
    news_df['content'] = news_df['content'].apply(stemming)

    # Features & Labels
    X = news_df['content'].values
    y = news_df['label'].values

    # Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=2
    )

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Accuracy
    train_acc = accuracy_score(
        model.predict(X_train),
        Y_train
    )

    test_acc = accuracy_score(
        model.predict(X_test),
        Y_test
    )

    return model, vectorizer, train_acc, test_acc

model, vectorizer, train_acc, test_acc = train_model()

st.title("📰 Fake News Detection System")
st.write("Detect whether a news article is **Real** or **Fake** using Machine Learning.")


with st.sidebar:
    st.header("📊 Model Information")

    st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
    st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")

    st.markdown("---")

    st.write("### 🛠 Technologies Used")
    st.write("""
    - Streamlit
    - Scikit-learn
    - TF-IDF Vectorizer
    - Logistic Regression
    - NLTK
    """)

input_text = st.text_area(
    "✍ Enter News Article",
    height=250,
    placeholder="Paste news article here..."
)


def predict_news(news_text):

    processed = stemming(news_text)

    vector_input = vectorizer.transform([processed])

    prediction = model.predict(vector_input)[0]

    probability = model.predict_proba(vector_input)

    confidence = np.max(probability) * 100

    return prediction, confidence


if st.button("🔍 Analyze News"):

    if input_text.strip() == "":
        st.warning("⚠ Please enter some news text.")
    else:

        with st.spinner("Analyzing news article..."):
            time.sleep(1)

            result, confidence = predict_news(input_text)

        st.markdown("---")

        if result == 1:
            st.markdown(f"""
            <div class="result-box" style="background-color:#7f1d1d;color:white;">
                🚨 FAKE NEWS <br><br>
                Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="result-box" style="background-color:#14532d;color:white;">
                ✅ REAL NEWS <br><br>
                Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")


