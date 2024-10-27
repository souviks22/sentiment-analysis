import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
noise = "https?://[^\s]+|@([A-Za-z0-9_]{1,15})|[^A-Za-z0-9]+"

def noisefree(text: str) -> str:
    alphanum = re.sub(noise,' ',text.lower())
    tokens = [stemmer.stem(word) for word in alphanum.split() if word not in stopwords]
    return ' '.join(tokens)

def normalization(X_train: pd.Series, y_train: pd.Series) -> tuple[Tokenizer, LabelEncoder]:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    return tokenizer, encoder

def get_splits(data: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, Tokenizer]:
    df = pd.read_csv(data,names=['label','id','date','status','user','text'])
    df = df[['text','label']]
    df['text'] = df['text'].apply(noisefree)

    training_set, testing_set = train_test_split(df,test_size=0.1)
    tokenizer, encoder = normalization(training_set['text'],training_set['label'])
    
    X_train = tokenizer.texts_to_sequences(training_set['text'])
    X_test = tokenizer.texts_to_sequences(testing_set['text'])
    X_train = pad_sequences(X_train,maxlen=35)
    X_test = pad_sequences(X_test,maxlen=35)
    
    y_train = encoder.transform(training_set['label'])
    y_test = encoder.transform(testing_set['label'])
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    return X_train, y_train, X_test, y_test, df['text'], tokenizer