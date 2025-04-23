import streamlit as st
import pickle
import numpy as np

# Load trained model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("🧠 Sentiment Analysis App")
st.write("Enter a message below to analyze its sentiment:")

user_input = st.text_area("Message", "")

if st.button("Analyze"):
    if user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        st.success(f"🎯 Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text.")
