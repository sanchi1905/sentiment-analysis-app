# 🧠 Sentiment Analysis App

A powerful **three-class sentiment analysis** web application built with Streamlit and machine learning. Analyzes text and classifies it as **Positive**, **Negative**, or **Neutral** with detailed probability breakdowns.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Three-Class Classification**: Positive, Negative, and Neutral sentiment detection
- **82.3% Accuracy**: Trained on 10,000+ sentiment-labeled tweets
- **Interactive UI**: Beautiful Streamlit interface with visual probability bars
- **Real-time Analysis**: Instant sentiment predictions with confidence scores
- **Advanced NLP**: Negation handling, n-gram features, and balanced training
- **Detailed Insights**: Confidence indicators and preprocessed text visualization

## 🚀 Live Demo

Try the app live: **[Sentiment Analysis App](https://your-app-name.streamlit.app)** *(Update after deployment)*

## 📊 Model Performance

```
Overall Accuracy: 82.3%

Class-wise Metrics:
├─ Positive:  78% precision, 83% recall, 80% F1
├─ Negative:  88% precision, 94% recall, 90% F1
└─ Neutral:   84% precision, 74% recall, 79% F1
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **NLP**: NLTK
- **Model**: Logistic Regression with TF-IDF (unigrams, bigrams, trigrams)
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Quick Start (Local)

1. **Clone the repository**
```bash
git clone https://github.com/sanchi1905/sentiment-analysis-app.git
cd sentiment-analysis-app
```

2. **Create virtual environment and install dependencies**
```bash
# Windows
bootstrap.cmd

# Or manually:
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Download NLTK data**
```bash
python ensure_nltk_data.py
```

4. **Train the model** (optional - pre-trained model included)
```bash
python train_3class.py
```

5. **Run the app**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 🎯 Usage Examples

**Positive Example:**
```
Input: "I am absolutely thrilled and excited about my wonderful life!"
Output: Positive (99.5% confidence)
```

**Negative Example:**
```
Input: "I feel hopeless and depressed, nothing matters anymore."
Output: Negative (58.2% confidence)
```

**Neutral Example:**
```
Input: "The meeting is scheduled for 3pm tomorrow."
Output: Neutral (82.7% confidence)
```

## 📁 Project Structure

```
sentiment-analysis-app/
├── app.py                          # Main Streamlit application
├── train_3class.py                 # Training script for 3-class model
├── model_pipeline_3class.pkl       # Trained model (generated)
├── sentiment_tweets3.csv           # Training dataset
├── requirements.txt                # Python dependencies
├── .streamlit/config.toml          # Streamlit configuration
├── tests/                          # Unit tests
├── QUICK_START.md                  # Quick start guide
└── IMPLEMENTATION_SUMMARY.md       # Technical documentation
```

## 🔧 Advanced Usage

### Retrain the Model

```bash
python train_3class.py
```

This will:
- Load and preprocess the dataset
- Derive neutral class using TextBlob polarity
- Train a LogisticRegression model with balanced class weights
- Save the model to `model_pipeline_3class.pkl`

### Run Tests

```bash
# Quick prediction tests
python test_3class.py

# Detailed accuracy demo
python demo_accuracy.py

# Unit tests
pytest tests/
```

### Docker Deployment

```bash
docker build -t sentiment-app .
docker run -p 8501:8501 sentiment-app
```

## 🌐 Deploy to Streamlit Cloud

1. **Fork/Push this repo to your GitHub**
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Click** "New app"
4. **Select** your repository and branch
5. **Set** main file path: `app.py`
6. **Click** "Deploy"

The app will automatically:
- Install dependencies from `requirements.txt`
- Download NLTK data
- Load the pre-trained model
- Start serving predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sanchi**
- GitHub: [@sanchi1905](https://github.com/sanchi1905)

## 🙏 Acknowledgments

- Dataset: Sentiment140 (Twitter sentiment data)
- Framework: Streamlit for the amazing web framework
- ML: scikit-learn for robust machine learning tools

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{sentiment_analysis_app,
  author = {Sanchi},
  title = {Three-Class Sentiment Analysis App},
  year = {2025},
  url = {https://github.com/sanchi1905/sentiment-analysis-app}
}
```

---

Made with ❤️ using Streamlit & scikit-learn
