import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report
import re

def load_data(database_filepath):
    """Load data from the database and create dataframe containing the messages, labels and their names 
    
    Args:
    database_filepath : Database where the data are saved
        
    Return : 3 dataframe X for the messages to predict, Y for their target labels and categories containing the labels names"""
    
    #create an engine to enable the connection to the database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    #read the data from the database and create the relevant dataframe
    df = pd.read_sql_query('SELECT * FROM DisasterResponse', engine)
    #dataframe containing the messages
    X = df['message']
    #dataframe containing the labels of each messages
    Y = df[['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']]
    #the name of the labels previously extracted
    category_names = Y.columns.values
    
    return X, Y, category_names


def tokenize(text):
    """tokenize text 
    
    Args:
    text : the text to tokenize
        
    Return : clean tokens of the text provided"""
    
    #save methods under variable to enable tokenizing the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    #list where the tokens will be saved into
    clean_tokens = []
    #llop through the texts to tokenize each messages
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Machine Learning pipeline containing the main parameters to use and creating the model
    
    Args:
    text : the text to tokenize
        
    Return : model of our Machine Learning program"""
    
    #creation of the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    

    #definition of the parameters for model's optimization
    parameters = {
        'clf__estimator__n_estimators':[5]
      
    }
    #idenfification of the best parameters to use for the model
    model = GridSearchCV(pipeline, parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Machine Learning model's evaluation
    
    Args:
    model : the model to use for our prediction
    X_test : the message for which the prediction needs to occur
    Y_test : the predicted labels for each messages
    category_names : the names of each label predicted
        
    Return : a classification report providing the accuracy and scores of our created model"""
    #Creating a prediction based on the test data
    Y_pred = model.predict(X_test)
    #creating the score of our prediction based on our target test labels
    classification_report(Y_test, Y_pred,target_names=category_names)
    
    return print(classification_report(Y_test, Y_pred,target_names=category_names))

def save_model(model, model_filepath):
    """Saving the model for future application
    
    Args:
    model : the model to use for our prediction
    model_filepath : define the place where the model needs to be saved
    """
    #creating a pickle file containing our model
    import pickle as pk
    filename = 'models/disaster_model.pkl'
    #saving the file into the defined path
    pk.dump(model, open(filename, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
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