from tensorflow.keras.models import load_model
from scripts.preprocessing import normalize

model = load_model("models/sentiment.keras")
threshold = 0.5

def infer(text: str) -> str:
    vectorized = normalize(text)
    probability = model.predict(vectorized)
    sentiment = 'positive' if probability >= 0.5 else 'negative'
    return sentiment