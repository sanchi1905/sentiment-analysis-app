import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# ===========================
# Load pre-trained 3-class model pipeline
# ===========================
try:
    model_pipeline = joblib.load('model_pipeline_3class.pkl')
except FileNotFoundError:
    st.error("Model file 'model_pipeline_3class.pkl' not found. Please ensure the model is trained.")
    st.stop()

# Ensure stopwords are downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ===========================
# Text Preprocessing Functions
# ===========================
def clean_text(text):
    """
    Clean and preprocess text with negation handling.
    """
    text = text.lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Handle negations by attaching NOT_ prefix
    negation_words = ['not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                      'nowhere', 'none', "n't", 'cannot', 'cant', "won't", 
                      "wouldn't", "shouldn't", "couldn't", "doesn't", "don't",
                      "didn't", "isn't", "aren't", "wasn't", "weren't"]
    
    words = text.split()
    negated = False
    processed_words = []
    
    for word in words:
        # Remove punctuation from word
        clean_word = re.sub(r'[^a-z\s]', '', word)
        
        if clean_word in negation_words:
            negated = True
            continue
        
        if negated and clean_word:
            processed_words.append(f"NOT_{clean_word}")
            negated = False
        elif clean_word:
            processed_words.append(clean_word)
    
    # Remove stopwords (but keep negated words)
    stop_words = set(stopwords.words('english'))
    words = [word for word in processed_words if word.startswith('NOT_') or word not in stop_words]
    
    return ' '.join(words)

def create_probability_bar(label, probability, color):
    """
    Create a styled probability bar using Streamlit markdown and HTML.
    """
    percentage = probability * 100
    bar_html = f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: bold;">{label}</span>
            <span style="font-weight: bold;">{percentage:.1f}%</span>
        </div>
        <div style="background-color: #e0e0e0; border-radius: 5px; overflow: hidden;">
            <div style="background-color: {color}; width: {percentage}%; height: 25px; border-radius: 5px; 
                        transition: width 0.3s ease;"></div>
        </div>
    </div>
    """
    return bar_html

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(
    page_title="Sentiment Analysis - 3 Class Prediction",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter any text to analyze its sentiment: Positive, Negative, or Neutral</div>', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Enter your text here:",
        height=150,
        placeholder="Type or paste any text to analyze its sentiment..."
    )

with col2:
    st.markdown("### 📊 Model Info")
    st.info("""
    **Model:** Logistic Regression  
    **Features:** TF-IDF (1-3 grams)  
    **Accuracy:** 82.3%  
    **Classes:** 
    - 😊 Positive
    - 😐 Neutral  
    - 😞 Negative
    """)

# Predict button
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            # Preprocess the input
            cleaned_input = clean_text(user_input)
            
            # Predict sentiment using the pipeline
            prediction = model_pipeline.predict([cleaned_input])[0]
            probabilities = model_pipeline.predict_proba([cleaned_input])[0]
            
            # Map prediction to label
            class_labels = {0: "Positive", 1: "Negative", 2: "Neutral"}
            sentiment = class_labels[prediction]
            
            # Display results
            st.markdown("---")
            st.markdown("## 📈 Analysis Results")
            
            # Show the preprocessed text
            with st.expander("🔍 View Preprocessed Text"):
                st.code(cleaned_input)
            
            # Determine emoji and color based on prediction
            emoji_map = {0: "😊", 1: "😞", 2: "😐"}
            color_map = {0: "#28a745", 1: "#dc3545", 2: "#ffc107"}
            
            # Main result with larger emoji
            st.markdown(f"### Predicted Sentiment: {emoji_map[prediction]} **{sentiment}**")
            
            # Probability bars for all three classes
            st.markdown("### Confidence Breakdown:")
            
            positive_bar = create_probability_bar("Positive 😊", probabilities[0], "#28a745")
            negative_bar = create_probability_bar("Negative 😞", probabilities[1], "#dc3545")
            neutral_bar = create_probability_bar("Neutral 😐", probabilities[2], "#ffc107")
            
            st.markdown(positive_bar, unsafe_allow_html=True)
            st.markdown(negative_bar, unsafe_allow_html=True)
            st.markdown(neutral_bar, unsafe_allow_html=True)
            
    else:
        st.warning("⚠️ Please enter some text to analyze.")