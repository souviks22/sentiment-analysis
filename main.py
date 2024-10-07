from scripts.preprocessing import get_splits
from scripts.embedding import get_embedding_values
from scripts.training import get_model
from scripts.evaluation import evaluate

def main():
    X_train, y_train, X_test, y_test, corpus, tokenizer = get_splits('data/tweets.csv')
    embedding = get_embedding_values(corpus,tokenizer)
    model = get_model(X_train,y_train,X_test,y_test,embedding)
    evaluate(X_test,y_test,model)

if __name__ == '__main__':
    main()