import nltk
import sys

required = ["stopwords", "punkt"]
for pkg in required:
    try:
        nltk.data.find(f"corpora/{pkg}")
        print(f"NLTK: {pkg} already available")
    except LookupError:
        print(f"NLTK: {pkg} not found — downloading...")
        try:
            nltk.download(pkg)
            print(f"NLTK: {pkg} downloaded")
        except Exception as e:
            print(f"NLTK: failed to download {pkg}: {e}")
            sys.exit(2)

print("NLTK data check complete")
