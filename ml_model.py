import os
import nltk
import logging
from joblib import load
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

log = logging.getLogger('uvicorn')

def tokenizer_fucntion(text):
    tt = TweetTokenizer()
    tokens = tt.tokenize(text)
    return tokens

tokenizer = tokenizer_fucntion

class Model:
    def __init__(self):
        self.model = load(os.path.dirname(__file__)+ "/models/logistic_regression.joblib")

    def predict(self, text) -> list:
        log.info("Predicting...")
        prediction = self.model.predict([text])
        log.info(f"Input Prompt: {text}")
        log.info(f"Prediction: {prediction[0]}")
        return prediction[0]