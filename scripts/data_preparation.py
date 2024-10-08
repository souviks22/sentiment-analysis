from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(input_path):
  spark = SparkSession.builder.appName('TwitterSentimentAnalysis').getOrCreate()
  schema = StructType([
    StructField('label',IntegerType(),True),
    StructField('id',StringType(),True),
    StructField('date',StringType(),True),
    StructField('status',StringType(),True),
    StructField('user',StringType(),True),
    StructField('text',StringType(),True)
  ])
  data = spark.read.csv(input_path,schema=schema,header=False)
  data = data.sample(withReplacement=False,fraction=0.01)
  tokenizer = Tokenizer(inputCol='text',outputCol='words')
  words_data = tokenizer.transform(data)
  
  remover = StopWordsRemover(inputCol='words',outputCol='filtered_words')
  filtered_data = remover.transform(words_data)
  
  texts = [row['filtered_words'] for row in filtered_data.select('filtered_words').collect()]
  keras_tokenizer = KerasTokenizer(num_words=10000,oov_token='<OOV>')
  keras_tokenizer.fit_on_texts(texts)
  
  sequences = keras_tokenizer.texts_to_sequences(texts)
  padded_sequences = pad_sequences(sequences,maxlen=100,padding='post',truncating='post')
  labels = [row['label'] // 4 for row in filtered_data.select('label').collect()]
  labels = np.array(labels).reshape(-1,1)
  
  return padded_sequences, labels, keras_tokenizer

def split_data(data,labels):
  X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
  return X_train, X_test, y_train, y_test