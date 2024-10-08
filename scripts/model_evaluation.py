from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def evaluate_model(X_test,y_test):
  model = load_model('lstm_model')
  predictions = model.predict(X_test)
  predicted_labels = predictions.reshape(-1) >= 0.5
  true_labels = y_test.reshape(-1)
  accuracy = accuracy_score(true_labels,predicted_labels)
  print(f'Test Accuracy: {accuracy*100:.2f}%')