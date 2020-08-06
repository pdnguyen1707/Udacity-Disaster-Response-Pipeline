import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loading Messages and Categories from Destination Database
    
    Arguments: 
        messages_filepath -> Path to the CSV containing file messages
        categories_filepath -> Path to the CSV containing file categories
        
    Output:
        df -> Combined data containing messages and categories
        
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    
    return df


def clean_data(df):
    """
    Clean Data Function
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """
    
    categories = df.categories.str.split(';', expand = True) 
    categories.columns = categories.iloc[0].str.replace(r'[-0-9]', '')
    row = categories.iloc[0, :]
    categories.columns = categories.iloc[0].str.replace(r'[-0-9]','')
    for column in categories:
        categories[column] = categories[column].str.replace(r'[^0-9]','')
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(r'[^0-9]','')
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int32')
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates(subset = 'id')
    
    return df


def save_data(df, database_filename,table_name = 'StagingMLTable'):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table_name, engine, index=False)
    print("Data was saved to {} in the {} table".format(database_filename, table_name))
    



def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
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

