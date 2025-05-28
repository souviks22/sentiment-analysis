from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
import json
import numpy as np

stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
noise = 'https?://[^\s]+|@([A-Za-z0-9_]{1,15})|[^A-Za-z0-9]+'

def noisefree(text: str) -> str:
    alphanum = re.sub(noise,' ',text.lower())
    tokens = [stemmer.stem(word) for word in alphanum.split() if word not in stopwords]
    return ' '.join(tokens)

with open('models/sentiment_tokenizer.json') as f:
    tokenizer = tokenizer_from_json(json.load(f))

def normalize(text: str) -> np.ndarray:
    cleaned = noisefree(text)
    sequenced = tokenizer.texts_to_sequences(np.array([cleaned]))
    padded = pad_sequences(sequenced, maxlen=35)
    return padded