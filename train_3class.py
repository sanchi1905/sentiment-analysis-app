"""
Three-class sentiment analysis training script.
Classes: 0=Positive, 1=Negative, 2=Neutral

Strategy: Derive Neutral class from original data using TextBlob polarity
for messages with low absolute polarity scores.
"""
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from nltk.corpus import stopwords

# Ensure NLTK data
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

# Negation words for preprocessing
NEGATIONS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
             'hardly', 'scarcely', 'barely', "n't", 'cannot', "won't", "don't",
             "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "doesn't"}

# Neutral indicators (keywords that suggest neutral/factual content)
NEUTRAL_INDICATORS = {
    'okay', 'ok', 'fine', 'alright', 'whatever', 'meh', 'nothing special',
    'normal', 'average', 'usual', 'typical', 'regular', 'standard', 'moderate',
    'neutral', 'indifferent', 'neither', 'sort of', 'kind of', 'maybe', 'perhaps'
}

def clean_text(text):
    """Clean and preprocess text with negation handling."""
    if not isinstance(text, str):
        return ''
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.[^\s]+', '', text)
    # Remove mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove non-alphabetic
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    
    # Negation handling: prefix next token with NOT_
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


def derive_neutral_class(df):
    """
    Derive a neutral class from existing data using TextBlob polarity.
    Strategy:
    - Strong negative indicators (depression, suicide, hopeless) → stay class 1
    - Strong positive indicators (love, happy, great) → stay class 0
    - Neutral keywords or low polarity → class 2
    """
    from textblob import TextBlob
    
    # Strong negative keywords that should remain negative
    STRONG_NEGATIVE = {
        'suicid', 'kill', 'die', 'death', 'hopeless', 'worthless', 'depressed',
        'hate myself', 'want to die', 'end it', 'give up', 'no point', 'miserable'
    }
    
    # Strong positive keywords
    STRONG_POSITIVE = {
        'love', 'amazing', 'wonderful', 'fantastic', 'excellent', 'great',
        'blessed', 'grateful', 'excited', 'perfect', 'beautiful', 'awesome'
    }
    
    print("Deriving neutral class from original binary labels...")
    
    new_labels = []
    neutral_count = 0
    
    for idx, row in df.iterrows():
        text = str(row['message to examine'])
        original_label = row['label (depression result)']
        text_lower = text.lower()
        
        # Check for strong indicators
        has_strong_negative = any(kw in text_lower for kw in STRONG_NEGATIVE)
        has_strong_positive = any(kw in text_lower for kw in STRONG_POSITIVE)
        has_neutral_keyword = any(kw in text_lower for kw in NEUTRAL_INDICATORS)
        
        # Calculate polarity
        try:
            polarity = TextBlob(text).sentiment.polarity
        except:
            polarity = 0.0
        
        # Decision logic:
        if has_strong_negative:
            # Keep as negative
            new_labels.append(1)
        elif has_strong_positive:
            # Keep as positive
            new_labels.append(0)
        elif has_neutral_keyword or (abs(polarity) < 0.15 and original_label == 0):
            # Make neutral if has neutral keywords or low polarity for positive samples
            new_labels.append(2)
            neutral_count += 1
        else:
            # Keep original label
            new_labels.append(original_label)
    
    df['label_3class'] = new_labels
    print(f"Created {neutral_count} neutral samples from {len(df)} total samples")
    print("New label distribution:")
    print(df['label_3class'].value_counts().sort_index())
    
    return df


def balance_classes(X, y, target_ratio=0.7):
    """
    Balance classes using smart oversampling.
    Ensure minority classes have at least target_ratio of majority class size.
    """
    from collections import Counter
    class_counts = Counter(y)
    print(f"\nOriginal class distribution: {class_counts}")
    
    max_count = max(class_counts.values())
    target_count = int(max_count * target_ratio)
    
    X_balanced = []
    y_balanced = []
    
    for cls in sorted(class_counts.keys()):
        # Get all samples for this class
        indices = [i for i, label in enumerate(y) if label == cls]
        cls_X = [X[i] for i in indices]
        cls_y = [y[i] for i in indices]
        
        # Add original samples
        X_balanced.extend(cls_X)
        y_balanced.extend(cls_y)
        
        # Oversample if needed
        current_count = len(cls_X)
        if current_count < target_count:
            needed = target_count - current_count
            oversample_indices = np.random.choice(len(cls_X), size=needed, replace=True)
            X_balanced.extend([cls_X[i] for i in oversample_indices])
            y_balanced.extend([cls_y[i] for i in oversample_indices])
    
    final_counts = Counter(y_balanced)
    print(f"Balanced class distribution: {final_counts}")
    
    return X_balanced, y_balanced


def main():
    print("Loading dataset...")
    df = pd.read_csv('sentiment_tweets3.csv')
    print(f"Original dataset shape: {df.shape}")
    print(f"Original label distribution:\n{df['label (depression result)'].value_counts()}")
    
    # Derive neutral class
    df = derive_neutral_class(df)
    
    # Prepare data
    print("\nCleaning and preprocessing text...")
    X = [clean_text(text) for text in df['message to examine']]
    y = df['label_3class'].tolist()
    
    # Balance classes
    X, y = balance_classes(X, y, target_ratio=0.6)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Build pipeline with optimized parameters for 3-class classification
    print("\nBuilding and training model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=10000
        )),
        ('clf', LogisticRegression(
            solver='saga',  # Better for multiclass
            max_iter=2000,
            class_weight='balanced',
            multi_class='multinomial',
            random_state=42,
            C=1.5  # Regularization strength
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_pred = pipeline.predict(X_test)
    
    target_names = ['Positive', 'Negative', 'Neutral']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n(Rows: Actual | Columns: Predicted)")
    print("Order: Positive(0), Negative(1), Neutral(2)")
    
    # Save model
    model_path = 'model_pipeline_3class.pkl'
    joblib.dump(pipeline, model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Test with sample sentences
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    test_samples = [
        "I am so happy and excited about this wonderful day!",
        "I feel hopeless and want to die, nobody cares about me.",
        "The weather is okay today, nothing special.",
        "This is fine, I guess. Whatever.",
        "I love my life and I'm grateful for everything!",
        "I'm so depressed and tired of everything.",
        "It's alright, pretty normal day."
    ]
    
    for text in test_samples:
        cleaned = clean_text(text)
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0]
        
        print(f"\nText: {text}")
        print(f"Predicted: {target_names[pred]}")
        print(f"Probabilities: Positive={proba[0]:.3f}, Negative={proba[1]:.3f}, Neutral={proba[2]:.3f}")


if __name__ == '__main__':
    main()
