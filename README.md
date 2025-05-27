# Twitter Sentiment Analysis

This project performs **sentiment analysis on tweets** using a custom deep learning model with pre-trained word embeddings. It includes complete steps from raw data processing and visualization to training, evaluation, and saving the final model.

---

## Dataset

- **Source**: [Kaggle - Twitter Sentiment 140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Labels**: `0` for Negative, `4` for Positive (converted to `Negative`/`Positive`)

---

## Data Analysis & Visualization

Using `pandas` and `seaborn`:

- Distribution of sentiment classes
- Tweet length distribution
- Separate visualization for positive and negative tweet lengths

Example:
```python
sns.countplot(x='label', data=df)
sns.boxplot(y='length', data=df)
````

---

## Preprocessing

Steps included:

* Noise removal using regex (URLs, mentions, non-alphanumeric)
* Stopword removal
* Stemming using `SnowballStemmer`
* Tokenization with Keras
* Padding to uniform input length

```python
text = re.sub(r"https?://[^\s]+|@(\w+)|[^A-Za-z0-9]+", " ", text.lower())
```

---

## Word Embedding

* Trained a **custom Word2Vec** model (`Gensim`)
* Embedding vectors of dimension `100`
* Initialized embedding matrix for Keras with pre-trained vectors

---

## Model Architecture

* **Embedding Layer** (non-trainable, custom pretrained)
* **SpatialDropout1D**
* **1D Convolution Layer**
* **Bidirectional LSTM**
* **Dense Layers with Dropout**
* **Sigmoid Output for Binary Classification**

```python
input → Embedding → Dropout → Conv1D → BiLSTM → Dense → Dropout → Output
```

Loss Function: `binary_crossentropy`
Optimizer: `Adam` with `ReduceLROnPlateau`
Training Epochs: `10`
Batch Size: `512`

---

## Evaluation

The model was evaluated on a held-out test set of **160,000 tweets**.

| Metric                  | Value |
| ----------------------- | ----- |
| **Accuracy**            | 75%   |
| **F1 Score (Negative)** | 0.73  |
| **F1 Score (Positive)** | 0.76  |
| **Macro Avg F1**        | 0.75  |

The model performs slightly better on **positive sentiments** (recall: 82%) but is reasonably balanced overall.

---

## Model Persistence

* Trained model saved as `lstm_model.keras`
* Word embeddings saved as `word_embedding.model` (Gensim)
