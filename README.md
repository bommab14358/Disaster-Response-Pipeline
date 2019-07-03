# Disaster-Response-Pipeline
Create required data processing and machine learning pipeline to enable proper classification of the messages or social media mentions during disaster

## ETL Pipeline
### Cleaning the data
Import the messages data and their corresponding categories, process the categories dataset to 
 - Extract labels for all 36 categories
 - Create one column per category label
 - Convert the flag for each category label into binary

### Saving the data into SQL Database
Data after cleaning the category labels is merged with messages data and exported as a table into a SQL database

## Modelling Pipeline
### NLP Pipeline
NLP pipeline extracts model usable features from Messages -

 - Tokenizer:
   - Breaks the messages into words
   - Substitues URLs with placeholders
   - Converts all characters to lowercase
   - Lemmatizes the words i.e. removes the inflectional forms of the words
   
#### Feature Generation
  - TFIDF word vectorizer
    - Vectorizer converts the messages into word count vectors i.e. a vector with unique list of words in all the messages are created and the counts for each word are reported as vector data
    - TFIDF: Term Frequency and Inverse Document Frequency as a technique downweights the words that are more recurrent across all messages and overweights the words that are less prevalent
    - Output from the TFIDF word vectorization is filtered to include only top 1000 words

  - TFIDF N-gram vectorizer
    - N-grams are the sequence of characters that are prevalent in the messages dataset
    - Data is vectorized as N-grams and the frequency are adjusted for TFIDF
    - Output from TFIDF N-gram vectorization is filtered to include only top 
    
### Modelling Pipeline    
RandomForest Classifier is used for classifying the messages into the multi-label output i.e. one message belonging to more than one category. Pipeline outputs the Accuracy, Precision, Recall and F1 score for the overall model and for each category 

## Instructions:

 1.) Run the following commands in the project's root directory to set up your database and model.

 - To run ETL pipeline that cleans data and stores in database 
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 - To run ML pipeline that trains classifier and saves 
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

 2.) Run the following command in the app's directory to run your web app. 
    python run.py

 - Go to http://0.0.0.0:3001/
