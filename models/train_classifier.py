import sys
# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
#nltk.download()
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import re
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
nltk.download('stopwords')
#from custom_transformer import StartingVerbExtractor
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.base import BaseEstimator,TransformerMixin

import pickle

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])

import matplotlib.pyplot as plt



def load_data(database_filepath, table_name = 'StagingMLTable'):
    """
    Loading Data from Database Function
    
    Arguments:
        database_filepath -> Path to SQLite Destination database (e.g. DisasterResponse.db)
        table_name -> the name of table in the destination database
    
    Outputs:
        X -> A dataframe containing features
        Y -> A dataframe containing labels
        category_names -> List of categories
    """
    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(table_name,engine)
    df.drop('child_alone', axis = 1, inplace = True)
    df.drop('id', axis = 1, inplace = True)
    X = df.loc[:, 'message']
    y = df.iloc[:, 4:]
    category_names = y.columns.tolist()
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text messages needed to be tokenized
        
    Outputs:
        clean_tokens -> List of tokens extracted from provided text
        
    """
    
    #Normalize Text:
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
    
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def build_model():
    """
    Building pipeline function
    
    Output: Scikit ML Pipeline that processes text messages
    according to NLP best-practice and apply a classifier function.
    
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    #parameters = {'clf__estimator__n_estimators': [50, 100],
                  #'clf__estimator__min_samples_split': [2, 3, 4],
                  #'clf__estimator__criterion': ['entropy', 'gini']}
    #cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate the performance of model
    
    This function applies the model to test set and prints
    out the accuracy of the prediction of categories.
    
    """
    #Y_prediction_train = model.predict(X_train)
    Y_prediction_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_prediction_test, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    Save Model Function
    
    This Function saves trained model as Pickle file, to be loaded
    for prediction later. 
    
    Arguments: 
        model -> Scikit ML Pipeline and GridSearchCV
        model_filepath -> destination path to save .pkl file
        
    """
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    model_pkl.close()


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
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

    
    