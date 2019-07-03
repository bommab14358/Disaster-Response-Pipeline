import sys
import pandas as pd
import re
import nltk
import pickle

nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table('data', con = engine.connect())
    # load data from database
    X = df['message'].values
    Y = df.drop(['id','message', 'original', 'genre'], axis=1).values
    labels = df.drop(['id','message', 'original', 'genre'], axis=1).columns
    return X, Y, labels

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', FeatureUnion([
            ('tfidf_word', TfidfVectorizer(tokenizer=tokenize, analyzer='word', max_features=2000)),
            ('tfidf_ngram', TfidfVectorizer(tokenizer=tokenize, analyzer = 'char', ngram_range=(3,7), max_features=5000))
        ])),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    return pipeline
    


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    if scipy.sparse.issparse(y_pred):
        y_pred = y_pred.toarray()
    print('Overall model performance metrics.. \n')
    print('Accuracy: ', 100*round(accuracy_score(Y_test, y_pred), 4))
    print('Precision: ', 100*round(precision_score(Y_test, y_pred, average='micro'), 4))
    print('Recall: ', 100*round(recall_score(Y_test, y_pred, average='micro'), 4))
    print('F1 Score : ', round(f1_score(Y_test, y_pred, average = 'micro'), 4))
    results = []
    y_actual = pd.DataFrame(Y_test, columns = category_names)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    for i in category_names:
        results.append([100*round(accuracy_score(y_actual[i], y_pred[i]), 4),
                        100*round(precision_score(y_actual[i], y_pred[i]), 4),
                        100*round(recall_score(y_actual[i], y_pred[i]), 4),
                       round(f1_score(y_actual[i], y_pred[i]), 4)])
    results = pd.DataFrame(results, columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score'], index = category_names)
    return results

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
