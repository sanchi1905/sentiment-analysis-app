"""
train_improved.py

Train an improved pipeline: TfidfVectorizer(ngram_range=(1,2)) + LogisticRegression(class_weight='balanced')
Saves to: model_pipeline.pkl (overwrites existing)
"""
from pathlib import Path
import re
import sys
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

URL_RE = re.compile(r'http\S+|www\.[^\s]+')
NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s]')

# common negation words to handle
NEGATIONS = set(['not', "don't", "didn't", "isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "no"])


def ensure_nltk():
    try:
        stopwords.words('english')
    except Exception:
        nltk.download('stopwords')


def clean_text(text: str, stopwords_set):
    """
    Clean text and apply a simple negation handling: tokens following a negation
    word up to the next punctuation/stopword get joined with a prefix 'NOT_'.
    This is a light-weight approach to capture polarity flips (e.g. 'not good' -> 'NOT_good').
    """
    if not isinstance(text, str):
        return ''
    text = URL_RE.sub('', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # replace non-alpha with spaces (keep words)
    text = NON_ALPHA_RE.sub(' ', text)
    text = text.lower()

    tokens = text.split()
    out_tokens = []
    negate = False
    for tok in tokens:
        if tok in NEGATIONS:
            negate = True
            out_tokens.append(tok)
            continue
        if negate:
            # attach NOT_ prefix and turn off negation for the next content word
            nt = f'NOT_{tok}'
            out_tokens.append(nt)
            negate = False
        else:
            out_tokens.append(tok)

    # remove stopwords but keep NOT_ tokens
    words = [w for w in out_tokens if (w.startswith('NOT_') or w not in stopwords_set)]
    return ' '.join(words)


def main():
    data_path = Path('sentiment_tweets3.csv')
    out_path = Path('model_pipeline.pkl')

    ensure_nltk()
    STOPWORDS = set(stopwords.words('english'))

    print('Loading data...')
    df = pd.read_csv(data_path)
    if 'message to examine' not in df.columns:
        raise SystemExit("CSV must contain column 'message to examine'")

    df['clean_text'] = df['message to examine'].apply(lambda t: clean_text(t, STOPWORDS))
    X = df['clean_text']
    y = df['label (depression result)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Simple minority-class oversampling: duplicate minority rows until balance
    df_train = pd.DataFrame({'text': X_train, 'label': y_train})
    counts = df_train['label'].value_counts()
    if counts.min() / counts.max() < 0.5:
        # oversample minority class by simple duplication
        maj_class = counts.idxmax()
        min_class = counts.idxmin()
        maj_count = counts.max()
        min_df = df_train[df_train['label'] == min_class]
        reps = int(maj_count / len(min_df)) - 1
        extras = pd.concat([min_df] * reps, ignore_index=True) if reps > 0 else pd.DataFrame(columns=min_df.columns)
        df_train = pd.concat([df_train, extras], ignore_index=True)
        # shuffle
        df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train = df_train['text']
    y_train = df_train['label']

    print('Building improved pipeline...')
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'))
    ])

    print('Training improved model...')
    pipeline.fit(X_train, y_train)

    print('Evaluating...')
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    print('Saving improved pipeline...')
    joblib.dump(pipeline, out_path)
    print('Saved to', out_path)


if __name__ == '__main__':
    main()
