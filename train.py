"""
train.py

Lightweight training script that reproduces the notebook steps without Jupyter.
Outputs: model_pipeline.pkl (sklearn Pipeline: TfidfVectorizer + MultinomialNB)

Usage:
    python train.py --data sentiment_tweets3.csv --out model_pipeline.pkl

"""
import argparse
import re
import sys
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def ensure_nltk():
    try:
        stopwords.words('english')
    except Exception:
        nltk.download('stopwords')


URL_RE = re.compile(r'http\S+|www\.[^\s]+')
NON_ALPHA_RE = re.compile(r'[^a-zA-Z\s]')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = URL_RE.sub('', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = NON_ALPHA_RE.sub(' ', text)
    text = text.lower()
    words = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(words)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'message to examine' not in df.columns:
        raise ValueError("CSV must contain column 'message to examine'")
    return df


def train(args):
    data_path = Path(args.data)
    out_path = Path(args.out)

    ensure_nltk()
    global STOPWORDS
    STOPWORDS = set(stopwords.words('english'))

    print(f'Loading data from {data_path}...')
    df = load_data(data_path)

    print('Cleaning text...')
    df['clean_text'] = df['message to examine'].apply(clean_text)

    X = df['clean_text']
    y = df['label (depression result)']

    print('Splitting...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Building pipeline...')
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])

    print('Training...')
    pipeline.fit(X_train, y_train)

    print('Evaluating...')
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    print(f'Saving pipeline to {out_path}...')
    joblib.dump(pipeline, out_path)
    print('Done.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='sentiment_tweets3.csv', help='path to CSV')
    p.add_argument('--out', default='model_pipeline.pkl', help='output model path')
    return p.parse_args()


if __name__ == '__main__':
    STOPWORDS = set()
    args = parse_args()
    try:
        train(args)
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
