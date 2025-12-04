from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import app


def test_model_pipeline_smoke():
    # Create a tiny pipeline and attach to app to simulate a loaded model
    X = ["i love this", "i hate this"]
    y = [1, 0]
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    pipeline.fit(X, y)

    # attach to app (simulate model loaded at runtime)
    app.model_pipeline = pipeline

    pred = app.model_pipeline.predict(["i really love this!"])[0]
    assert int(pred) == 1
