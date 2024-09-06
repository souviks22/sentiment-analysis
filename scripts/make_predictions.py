from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_new_data(text,tokenizer):
  model = load_model('lstm_model.keras')
  sequence = tokenizer.texts_to_sequences([text])
  padded_sequence = pad_sequences(sequence,maxlen=100,padding='post',truncating='post')
  prediction = model.predict(padded_sequence)
  predicted_label = prediction.argmax(axis=1)[0]
  sentiment = {0:'Negative',1:'Positive'}[predicted_label]
  print(f'{text}: {sentiment}')