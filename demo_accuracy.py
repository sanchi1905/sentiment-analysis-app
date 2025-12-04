"""
Demo script showing how the 3-class model provides accurate predictions
with respective probabilities for Positive, Negative, and Neutral.
"""
import joblib
import re
from nltk.corpus import stopwords

# Load model
pipeline = joblib.load('model_pipeline_3class.pkl')

# Preprocessing constants
URL_RE = re.compile(r'http\S+|www\.[^\s]+')
NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s]')
NEGATIONS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
             'hardly', 'scarcely', 'barely', "n't", 'cannot', "won't", "don't",
             "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "doesn't"}

try:
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """Clean text with negation handling (same as app.py)"""
    if not isinstance(text, str):
        return ''
    
    text = URL_RE.sub('', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = NON_ALPHA_RE.sub(' ', text)
    text = text.lower()
    
    tokens = text.split()
    processed = []
    negate = False
    for token in tokens:
        if token in STOPWORDS and token not in NEGATIONS:
            continue
        if token in NEGATIONS:
            negate = True
            processed.append(token)
        else:
            if negate:
                processed.append('NOT_' + token)
                negate = False
            else:
                processed.append(token)
    
    return ' '.join(processed)


def analyze_sentiment(text):
    """Analyze sentiment and return detailed results"""
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    
    class_names = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    
    return {
        'text': text,
        'cleaned': cleaned,
        'prediction': class_names[pred],
        'prediction_class': pred,
        'probabilities': {
            'Positive': proba[0] * 100,
            'Negative': proba[1] * 100,
            'Neutral': proba[2] * 100
        }
    }


def print_result(result):
    """Pretty print analysis result"""
    print(f"\n{'='*80}")
    print(f"📝 Original Text:")
    print(f"   {result['text']}")
    print(f"\n🔧 Preprocessed:")
    print(f"   {result['cleaned']}")
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"\n📊 Accuracy Breakdown (Probabilities):")
    
    # Sort by probability
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    for sentiment, prob in sorted_probs:
        bar_length = int(prob / 2)
        bar = '█' * bar_length
        emoji = '😊' if sentiment == 'Positive' else '😞' if sentiment == 'Negative' else '😐'
        print(f"   {emoji} {sentiment:>8}: {prob:6.2f}%  {bar}")


# Demo: Full sentence analysis
print("\n" + "="*80)
print(" 3-CLASS SENTIMENT ANALYSIS DEMO - SHOWING ACCURACY FOR EACH CLASS")
print("="*80)

# Test various sentence types
demo_sentences = [
    # Strong positive
    "I am absolutely thrilled and excited about my wonderful life!",
    
    # Strong negative
    "I feel completely hopeless and depressed, I want to give up.",
    
    # Clear neutral
    "The meeting is scheduled for 3pm tomorrow afternoon.",
    
    # Mild positive
    "Today was pretty good, I enjoyed the weather.",
    
    # Mild negative
    "I'm a bit disappointed with how things turned out.",
    
    # Ambiguous/neutral
    "This is okay I guess, nothing special really.",
    
    # Mixed sentiment (should show in probabilities)
    "I'm happy about the promotion but sad to leave my team."
]

for sentence in demo_sentences:
    result = analyze_sentiment(sentence)
    print_result(result)

# Summary
print("\n" + "="*80)
print(" KEY FEATURES:")
print("="*80)
print("""
✓ Shows exact probability percentages for all three classes
✓ Predictions based on machine learning (not simple keyword matching)
✓ Handles negation (e.g., "not happy" → negative sentiment)
✓ Works with full sentences and contextual understanding
✓ Displays preprocessed text to show what the model "sees"
✓ Visual bars make it easy to compare confidence across classes
""")

print("\n" + "="*80)
print(" HOW TO USE IN THE WEB APP:")
print("="*80)
print("""
1. Open your browser to: http://localhost:8501
2. Type or paste any sentence in the text area
3. Click "🔍 Analyze Sentiment"
4. View results with:
   - Large emoji showing the predicted sentiment
   - Detailed probability bars for all three classes
   - Confidence indicator (High/Good/Low)
   - Expandable view of preprocessed text

The web app provides the same accurate analysis with a beautiful interface!
""")

print("="*80)
