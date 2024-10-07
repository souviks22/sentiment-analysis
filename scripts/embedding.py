import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer

model = {}

def train_word_embedding(corpus: pd.Series) -> Word2Vec:
    corpus = corpus.apply(lambda text: text.split())
    embedding_model = Word2Vec(sentences=corpus,vector_size=100,window=5,min_count=1,workers=5)
    embedding_model.save('models/word_embedding.model')
    model['vector'] = embedding_model.wv
    return model['vector']

def get_embedding_values(corpus: pd.Series, tokenizer: Tokenizer) -> np.ndarray:
    vector = model['vector'] if 'vector' in model else train_word_embedding(corpus)
    embedding = [vector[word] if word in vector else [0]*100 for word in tokenizer.word_index]
    embedding.insert(0,[0]*100)
    return np.array(embedding)