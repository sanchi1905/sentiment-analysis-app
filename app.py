import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocesing function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(words)

#UI layout
st.set_page_config(page_title="Sentiment Analysis", page_icon=":smiley:", layout="centered")

st.markdown("<h1 style='text-align:center;'>🧠 Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("Enter a sentence or phrase and get its sentiment:")

# User input
user_input = st.text_input("Enter your message here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")