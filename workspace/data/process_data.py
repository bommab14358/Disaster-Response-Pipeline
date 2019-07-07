# Importing libraries
import sys
import pandas as pd                                # Pandas for basic dataframe manipulations
import nltk                                        # For Natural Language Processing
from sqlalchemy import create_engine               # To create SQL database to store the processed data
import sqlite3                                     # SQL compiler to interact with SQL databases 

def load_data(messages_filepath, categories_filepath):
    '''
    Objective: To import the messages and categories data that make up the independent and dependent variable respectively and to 
    create a merged dataset that contains all input information.
    
    Input -
    messages_filepath: Path to the location of messages data (including the filename)
    categories_filepath: Path to the location of categories data (including the filename)
    
    Output - 
    df: Dataframe containing both messages and categories data merged together by the "ID"
    '''
    # Read messages data into a dataframe and remove duplicates 
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop_duplicates()
    # Read categories data into a dataframe and remove duplicates
    categories = pd.read_csv(categories_filepath)
    categories = categories.drop_duplicates()
    # Add categories information to messages data by their ID, remove duplicates and return the final merged dataset
    df = messages.merge(categories, how = 'left', on = ['id'])
    df = df.drop_duplicates()
    return df


def clean_data(df):
    '''
    Objective: To clean and process the categories data to create individual variable for all 36 categories
    
    Input - 
    df - Dataframe created by merging "categories" data with "messages" data
    
    Output - 
    df - Dataframe with "categories" column replaced by 36 binary (only 1 or 0 as column values) columns i.e. one column each for 
    all 36 categories
    '''
    # Extract id and categories from the merged dataset
    categories = df[['id','categories']].copy()
    # Split "categories" into 36 individual category columns and extract category names from first row
    categories1 = categories.categories.str.split(';', expand = True)
    row = categories1.loc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories1.columns = category_colnames
    # Convert category columns into binary flags i.e. having only 0 or 1 as column values
    for column in categories1:
        categories1[column] = categories1[column].apply(lambda x: x.split('-')[1])
        categories1[column] = categories1[column].astype(int)
        categories1[column] = categories1[column].apply(lambda x: 1 if x>0 else 0)
    # Add id to the category columns
    categories1['id'] = categories['id'].copy()
    # Remove categories column and add the individual category columns to the merged dataset and return the updated merged dataset
    df = df.drop('categories', axis = 1)
    df = df.merge(categories1, how = 'left', on = ['id'])
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    Objective: To save the processed dataset into a SQL database
    
    Input -
    df: Dataframe containing the processed data after cleaning the categories column
    database_filename: Path to the location of SQL database (including database name)
    '''
    # Create SQL database at the location and create SQL engine to connect to the database
    engine = create_engine('sqlite:///'+database_filename)
    # Save the processed data as a table in the SQL database
    df.to_sql('data', engine, index=False, if_exists = 'replace')


def main():
    '''
    Function that executes all the steps of data processing pipeline in sequence 
    '''
    # Check if all required parameters are provided and show error message in the alternate case
    if len(sys.argv) == 4:
        # Parse all input parameters and save into parameter variables
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Import the messages and categories data using load_date module
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # Process categories data using the clean_data module
        print('Cleaning data...')
        df = clean_data(df)
        
        # Save the processed data into the SQL database using the save_data module
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
