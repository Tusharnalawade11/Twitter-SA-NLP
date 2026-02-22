import numpy as np
import pickle 
import re
import contractions

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



## Load the trained LSTM model and tokenizer
model = load_model('models/sentiment_lstm_model.keras')
tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))

stop_words = [word for word in stopwords.words('english') if word not in {'not', 'no', 'nor'}]


def text_preprocessing(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)

    text = contractions.fix(text)
    return text


# Function to preprocess input text
def preprocess_text(text):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences to the same length as the training data
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed
    return padded_sequences

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    # Use direct call instead of model.predict() for faster single-sample inference
    prediction = model(preprocessed_text, training=False)
    predicted_class = np.argmax(prediction.numpy(), axis=1)[0]
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[predicted_class]