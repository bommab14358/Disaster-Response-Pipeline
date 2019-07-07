# Importing necessary libraries
import pandas as pd         # For dataframe processing
import re                   # For regular experession processing
import nltk                 # For Natural Language Processing
import pickle               # For saving the final models 
import scipy                # For processing the sparse matrix
import time                 # For evalulating time taken in model training
import sys              
import warnings             # For suppressing warnings

warnings.filterwarnings('ignore')
nltk.download(['punkt', 'wordnet'])

from sqlalchemy import create_engine    # To create SQL engine to read the processed data
from nltk.stem import WordNetLemmatizer # To lemmatize the words extracted from the messages data 
# To vectorize the message text into words and apply TFIDF factorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer  
# To evaulate model performance 
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, accuracy_score
# To split the input data into train and test sets and for hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV

# To create NLP and modelling pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
# To create multi-label classifiers
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
# To classify the data into relevant classes
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    '''
    Objective: To import the data as dataframe from SQL database
    
    Input -
    database_filepath: Path to the location of SQL database (including database name)
    
    Output -
    X - Array containing message text which make up the independent variables
    Y - Array containing binary flag columns for each of 36 categories
    labels - Array containing names for all 36 categories
    '''
    # Create a SQL engine and connection to interact with the SQL database and import data as a dataframe
    engine = create_engine('sqlite:///'+database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table('data', con = engine.connect())
    # Save data in 'message' column as X variable
    X = df['message'].values
    # Save all 36 category columns as Y variable
    Y = df.drop(['id','message', 'original', 'genre'], axis=1).values
    # Extract column names for category columns and save as labels array
    labels = df.drop(['id','message', 'original', 'genre'], axis=1).columns
    return X, Y, labels

def tokenize(text):
    '''
    Objective: To split the message text into words and process the words into a model interpretable format
    
    Input -
    text: Message as a text item
    
    Output -
    clean_tokens: Array containing all the words that make up the text with its inflected forms removed and case lowered 
    '''
    # Define regular expression to identify URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Identify all the URLs within the text and replace the url with text 'urlplaceholder' 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Convert text into an array of words 
    tokens = nltk.word_tokenize(text)
    # Convert the words into lowercase and remove the inflection forms from the words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    # Return the array of processed words
    return clean_tokens


def build_model():
    '''
    Objective: Create a Feature engineering, modelling and hyperparameter tuning pipelines
    
    Input - None
    Output - 
    pipeline: Hyperparameter tuner like GridSearchCV which will identify the best parameters for the feature engineering and 
    modelling pipeline  
    '''
    # Create features by extracting words and ngrams from the text, factorizing them by TFIDF and selecting the top n features
    # Use RandomForest as classifier to identify the multiple labels that are associated with each message text
    pipeline = Pipeline([
        ('vect', FeatureUnion([
            ('tfidf_word', TfidfVectorizer(tokenizer=tokenize, analyzer='word', max_features=2000)),
            ('tfidf_ngram', TfidfVectorizer(tokenizer=tokenize, analyzer = 'char', ngram_range=(3,7), max_features=5000))
        ])),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    {'clf__n_estimators':[100, 200], 
             'vect__tfidf_word__max_features': [2000, 5000],
             'vect__tfidf_ngram__ngram_range': [(3, 7), (3, 9)]}
    gscv = GridSearchCV(fin_pipeline, param_grid=parameters, verbose = True, n_jobs = -1, cv = 2)
    return gscv
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Objective: To evaluate the performance of prediction pipeline on data it has not seen while training
    
    Input - 
    model: Prediction pipeline trained on subset of total available data (train data)
    X_test: Array containing messages that are not used for training the prediction pipeline
    Y_test: Array containing categories for all messages part of X_test dataset
    category_names: Array containing category labels
    
    Output-
    Overall model Accuracy, Precision, Recall and F1 score followed by the same set of metrics for each category
    '''
    # Calculate model predictions for messages in X_test set
    y_pred = model.predict(X_test)
    # Create an array of predictions if the output from previous step is a sparse matrix 
    if scipy.sparse.issparse(y_pred):
        y_pred = y_pred.toarray()
    # Calculate and print the Overall model Accuracy, Precision, Recall and F1 score
    print('Overall model performance metrics.. \n')
    print('Accuracy: ', 100*round(accuracy_score(Y_test, y_pred), 4))
    print('Precision: ', 100*round(precision_score(Y_test, y_pred, average='micro'), 4))
    print('Recall: ', 100*round(recall_score(Y_test, y_pred, average='micro'), 4))
    print('F1 Score : ', round(f1_score(Y_test, y_pred, average = 'micro'), 4))
    # Calculate category level performance metrics by using the predicted and actual labels by looping the performance calculation
    # for each category
    results = []
    y_actual = pd.DataFrame(Y_test, columns = category_names)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    for i in category_names:
        results.append([100*round(accuracy_score(y_actual[i], y_pred[i]), 4),
                        100*round(precision_score(y_actual[i], y_pred[i]), 4),
                        100*round(recall_score(y_actual[i], y_pred[i]), 4),
                       round(f1_score(y_actual[i], y_pred[i]), 4)])
    results = pd.DataFrame(results, columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score'], index = category_names)
    # Print category level performance metrics
    print(results)

def save_model(model, model_filepath):
    '''
    Objective: To save the final model as pickle file to be used in WebApp or to be used in later point of time
    
    Input -
    model: Model or prediction pipeline that is trained on the subset of the total available data
    model_filepath: Path to the location of pickle file that holds final model (including pickle file name)
    '''
    # To export the model or prediction pipeline as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Objective: To execute all steps for prediction pipeline creation in sequence
    '''
    # To check if all required parameters are passed to prediction pipeline creation
    if len(sys.argv) == 3:
        # Parse all input parameters into different variables
        database_filepath, model_filepath = sys.argv[1:]
        # To import the processed data for prediction pipeline creation by using the load_data module
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # To split the imported data into Train and Test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Instantiate the creation of feature engineering, modelling and Hyperparameter tuning pipeline by using build_model module
        print('Building model...')
        model = build_model()
        # To train the prediction pipeline on training data
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Extract the model with best hyperparemeters to be used for prediction
        model1 = model.best_estimator_
        
        # Evaluate best model's performance on test data by using evaluate_model module
        print('Evaluating model...')
        evaluate_model(model1, X_test, Y_test, category_names)
        
        # Save the best model as a pickle file by using save_model module
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model1, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
