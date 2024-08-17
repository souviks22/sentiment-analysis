from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

def build_model(vocab_size,input_length):
  model = Sequential([
    Embedding(input_dim=vocab_size,output_dim=64,input_length=input_length),
    LSTM(64),
    Dense(1,activation='sigmoid')
  ])

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model

def train_model(X_train,y_train):
  model = build_model(vocab_size=10000,input_length=100)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
  model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=10)
  model.save('lstm_model')
  return model