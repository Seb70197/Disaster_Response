import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """Load data containing the message and the categorization
    
    Args:
    messages_filepath : message for which the prediction needs to be made
    categories_filepath : the categories to predict
    
    Return : a dataframe merged from the 2 data sources"""
    
    #load the dataframe
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merging dataframe based on the "ID"
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """clean the dataframe loaded
    
    Args:
    df : the dataframe containing the text and categories
    
    Return : clean dataframe to use for building the model"""
    
    #Create a dataframe with all categories set as columns
    categories = df['categories'].str.split(';', expand=True)
    #extract the first row to define the names of the columns
    row = categories.iloc[0,:]
    
    #rename all columns based on the categories 
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    #ensure the presence of the values inside of the category columns
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    #drop the column categories containing all previously split columns
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    #remove duplicated values
    df.drop_duplicates(inplace=True)
    #replace the values which are not 1 or O from the column related
    df['related'] = df['related'].replace(2,1)
    
    return df


def save_data(df, database_filename):
    """enable connection to database and save the clean dataframe
    
    Args:
    df : the dataframe previously cleaned
    database_filename : database where to save the data
    
    Return : save the dataframe to the database"""
    from sqlalchemy import create_engine
    
    #create engine to enable connection to the database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    #save dataframe to the database
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')                         
    
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