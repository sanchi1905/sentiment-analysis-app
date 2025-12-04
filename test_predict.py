import joblib

model = joblib.load('model_pipeline.pkl')

samples = [
    "I am so happy and excited about today!",
    "I feel hopeless and don't want to get out of bed.",
    "This is okay, nothing special.",
]

print('Classes:', getattr(model, 'classes_', None))
for s in samples:
    X = [s]
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
    else:
        clf = model.named_steps.get('clf') if hasattr(model, 'named_steps') else None
        if clf is not None and hasattr(clf, 'predict_proba'):
            vec = model.named_steps.get('tfidf').transform(X)
            proba = clf.predict_proba(vec)[0]
    print('\nText:', s)
    print('Raw prediction:', pred)
    if proba is not None:
        print('Probabilities:', proba)
        # find index
        try:
            idx = list(model.classes_).index(pred)
            print('Pred prob:', proba[idx])
        except Exception:
            print('Pred prob (max):', max(proba))
    else:
        print('No predict_proba available')
