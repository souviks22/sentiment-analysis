from scripts.data_preparation import prepare_data, split_data
from scripts.model_training import train_model
from scripts.model_evaluation import evaluate_model
from scripts.make_predictions import predict_new_data

def main():
  input_path = 'data/tweets.csv'
  data, labels, tokenizer = prepare_data(input_path)
  X_train, X_test, y_train, y_test = split_data(data,labels)

  train_model(X_train,y_train)
  evaluate_model(X_test,y_test)

  text = 'I am so sad'
  predict_new_data(text,tokenizer)

if __name__ == '__main__':
  main()