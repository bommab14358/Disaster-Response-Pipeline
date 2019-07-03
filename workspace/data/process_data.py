import sys
import pandas as pd
import nltk
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop_duplicates()
    categories = pd.read_csv(categories_filepath)
    categories = categories.drop_duplicates()
    df = messages.merge(categories, how = 'left', on = ['id'])
    return df


def clean_data(df):
    categories = df[['id','categories']].copy()
    categories1 = categories.categories.str.split(';', expand = True)
    row = categories1.loc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories1.columns = category_colnames
    for column in categories1:
        categories1[column] = categories1[column].apply(lambda x: x.split('-')[1])
        categories1[column] = categories1[column].astype(int)
        categories1[column] = categories1[column].apply(lambda x: 1 if x>0 else 0)
    categories1['id'] = categories['id'].copy()
    df = df.drop('categories', axis = 1)
    df = df.merge(categories1, how = 'left', on = ['id'])
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    connection  = sqlite3.connect(database_filename)
    cursor = connection.cursor()
    cursor.execute('drop table if exists data')
    connection.close()
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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
