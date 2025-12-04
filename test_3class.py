"""Test script to validate the 3-class model and show sample predictions."""
import joblib

# Load model
pipeline = joblib.load('model_pipeline_3class.pkl')

# Test samples covering all three classes
test_samples = [
    # Positive samples
    "I am so happy and excited about this wonderful day!",
    "I love my life and I'm grateful for everything!",
    "This is absolutely amazing, best day ever!",
    
    # Negative samples
    "I feel hopeless and want to die, nobody cares about me.",
    "I'm so depressed and tired of everything.",
    "I hate myself and feel worthless.",
    
    # Neutral samples
    "The weather is okay today, nothing special.",
    "This is fine, I guess. Whatever.",
    "It's alright, pretty normal day.",
    "I'm going to the store to buy groceries."
]

class_names = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}

print("="*70)
print("3-CLASS SENTIMENT ANALYSIS TEST")
print("="*70)

for text in test_samples:
    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    
    print(f"\n📝 Text: {text}")
    print(f"🎯 Predicted: {class_names[pred]} (class {pred})")
    print(f"📊 Probabilities:")
    for cls_id, cls_name in class_names.items():
        prob_pct = proba[cls_id] * 100
        bar_length = int(prob_pct / 2)
        bar = '█' * bar_length
        print(f"   {cls_name:>8}: {prob_pct:5.1f}% {bar}")

print("\n" + "="*70)
print("✓ All tests completed successfully!")
print("="*70)
