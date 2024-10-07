# Twitter Sentiment Analysis
## Description
The project aims to create a efficient text recognition model developed using Natural Language Processing (NLP) priniciples that classifies a certain piece of text as positive or negative on basis of the sentiment experssed by the sentence. I used LSTM for text processing supported by Tensorflow.
## Training Dataset
The NLP model is trained on a large [Twitter dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) available on Kaggle consisting of 1.6m training examples which makes it experienced enough to further make wise decisions.
## Training Environment
The source code is written in a way such that we can perform the training as a Spark job over a distributed environment. Otherwise, the resources on a local machine is used for training. PySpark API is used to integrate this feature.
## Testing Results
Currently the whole dataset is not used to train the model as you can see in `analysis.ipynb` because of limited resources. The accuracy is about ~0.8 which can be further improved.
## Future Enhancements
I am thinking to develop a proper user interface to try it out more conveniently as soon as possible.
