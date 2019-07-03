# Disaster-Response-Pipeline
Create required data processing and machine learning pipeline to enable proper classification of the messages or social media mentions during disaster

## Cleaning the text data
Import the messages data and their corresponding categories, process the categories dataset to extract relevant category labels

## Instructions:
Run the following commands in the project's root directory to set up your database and model.

 - To run ETL pipeline that cleans data and stores in database 
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 - To run ML pipeline that trains classifier and saves 
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app. 
    python run.py

Go to http://0.0.0.0:3001/
