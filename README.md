# Twitter Sentiment Analysis using NLP & Deep Learning

This is a comprehensive data science project implementing sentiment analysis on Twitter data using both traditional NLP and deep learning techniques.

## Project Overview

This project analyzes Twitter posts and classifies them into three sentiment categories: **Positive**, **Neutral**, and **Negative**. It compares the performance of traditional machine learning models with modern deep learning approaches using LSTM networks.

## Features

### 1. **Traditional NLP Models**
- **Text Preprocessing**: URL removal, mention/hashtag filtering, lemmatization, and stopword removal
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) with 5,000 features
- **Models Implemented**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
- **Evaluation**: Classification reports, confusion matrices, and accuracy metrics

### 2. **Deep Learning with LSTM**
- **Text Processing**: Contraction expansion and advanced cleaning
- **Tokenization & Padding**: Converts text to sequences with fixed length (120 tokens)
- **Model Architecture**:
  - Embedding layer (120 dimensions)
  - LSTM layer (128 units) with return sequences
  - Dropout layer (50% rate) for regularization
  - LSTM layer (64 units)
  - Dense output layer (3 units, softmax activation)
- **Training**: Sparse categorical crossentropy loss with Adam optimizer

## Dataset

- **Source**: Twitter_Data.csv
- **Samples**: 2,000 tweets for traditional models, 62% (~3,100) for LSTM
- **Labels**: positive, neutral, negative

## Project Structure

```
Twitter-SA-NLP/
├── nb/
│   └── sentiment.ipynb          # Main Jupyter notebook
├── data/
│   └── Twitter_Data.csv         # Dataset
├── models/
│   ├── sa_logistic_model.pkl
│   ├── sa_svm_model.pkl
│   ├── sa_random_forest_model.pkl
│   ├── sa_tfidf_vectorizer.pkl
│   ├── sentiment_lstm_model.keras
│   └── tokenizer.pkl
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
contractions
tensorflow
keras
symspellpy (optional - for spell checking)
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk contractions tensorflow
```

## Usage

1. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook nb/sentiment.ipynb
   ```

2. **Execute cells sequentially** to:
   - Load and explore Twitter data
   - Preprocess text (cleaning, tokenization, lemmatization)
   - Train traditional models
   - Evaluate model performance
   - Train and evaluate LSTM model
   - Save trained models and tokenizers

## Model Performance

Each model produces:
- Classification reports (precision, recall, F1-score)
- Accuracy scores
- Confusion matrices with visualizations

## Key Results

- Traditional models trained on TF-IDF vectors
- LSTM captures sequential patterns in text
- Dropout prevents overfitting in deep learning model
- Models saved for future predictions

## Optimization Notes

For large datasets (10K+ rows):
- Skip spell checking (use `contractions.fix()` only)
- Use batch processing for text cleaning
- Keep sparse matrices instead of converting to dense arrays
- Implement data generators for LSTM training

## Author

Data Science Project - Boston Institute of Analytics

## License

MIT License
