import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Model

def evaluate(X_test: np.ndarray, y_test: np.ndarray, model: Model):
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1) >= 0.5
    y_test = y_test.reshape(-1)
    print(classification_report(y_test,predictions))