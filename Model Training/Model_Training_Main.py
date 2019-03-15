# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 07:14:33 2018

@author: binit kumar bhagat
"""
from classes import df_column_extractor, Converter
import pandas as pd
import numpy as np
import re, os
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#import matplotlib.pyplot as plt
from sklearn.externals import joblib

DIR = os.path.dirname(__file__)

def sentence_to_words(raw_sentence, drug_g=None):
    """This function modifies the sentences to words by removing html marks,
    non-letters, drug names, stopwords etc."""
    sent_text = BeautifulSoup(raw_sentence, "html.parser").get_text()  # Remove HTML markings
    letters = re.sub('[^a-zA-Z]', ' ', sent_text)  # Remove non-letters
    sent = letters.lower()
    if drug_g == drug_g:  # Check for nan values
        drug = re.sub('[^a-zA-Z1-9]', ' ', drug_g)
        sent = re.sub(str(drug.lower()), '', sent)  # Remove Drug Names
        words = sent.split()
    else:
        words = sent.split()
    stop_words = stopwords.words('english')
    important_words = [w for w in words if not w in stop_words]
    return ' '.join(important_words)

def stemming(text):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(w) for w in text]

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]


if __name__ == '__main__':
    
    ## Reading the Data
    data = pd.read_excel(DIR+'\Data For Model Training 10_Jul.xlsx', sheetname='Final')
    data.columns = ['Case ID','Drug Role','Trade Name','Generic Name','Text']
    print 'Data Shape = ',data.shape
    
    ## Preprocessing the text
    clean_text = []
    for i in range(len(data)):
        clean_text.append(sentence_to_words(data['Text'][i], data['Generic Name'][i]))
    stemmed_text = stemming(clean_text)
    lemmatized_text = lemmatization(clean_text)
    data['Clean_Text'], data['Stemmed'], data['Lemmatized'] = clean_text, stemmed_text, lemmatized_text
    
    ## Add Frequency column
    data['Frequency'] = np.ones(len(data), dtype=int)
    for i in range(len(data)):
        data['Frequency'][i] = int(len(data[(data['Case ID']==data['Case ID'][i]) & (data['Generic Name']==data['Generic Name'][i])]))
    
    ## Data Distribution
    print '\nOriginal Data Distribution: \n',data['Drug Role'].value_counts()
    
    ## Target Modification
    data['Drug Role'] = data['Drug Role'].replace('Suspect','Not Concomitant').replace('Treatment','Not Concomitant')
    print '\nFinal Data Distriution: \n', data['Drug Role'].value_counts()
    
    X = data.drop('Drug Role', axis=1)
    y = data['Drug Role']
    
    ## Stratified K Fold CV
    skf = StratifiedKFold(n_splits=5, random_state=111)
    index = 0
    for train_index, val_index in skf.split(X,y): 
        if index>3:
            break
        else:
            #print("Train:", train_index, "Validation:", val_index) 
            X_train, X_test = X.iloc[train_index], X.iloc[val_index] 
            y_train, y_test = y[train_index], y[val_index]
            index += 1
    
    ## Vectorizer and SVD Pipeline
    vec_svd_pipe = Pipeline([('stem', df_column_extractor('Stemmed')),
                         ('convert', Converter()),
                         ('vec', TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_df=0.8, max_features=3000)),
                         ('svd', TruncatedSVD(n_components=300, n_iter=10, random_state=222))
                        ])
    word_features = vec_svd_pipe.fit_transform(X_train)
    print '\nShape of the feature matrix: ', word_features.shape
    
    ## Classifiers
    forest = RandomForestClassifier(n_estimators=50, random_state=300)
    svm = SVC(C=0.4, gamma=0.1, kernel='linear', random_state=200, probability=True)
    model = forest
    
    ## Main Pipeline
    ### Frequency Feature added
    pipeline = Pipeline([('union', FeatureUnion([                     
                             ('vec_svd', vec_svd_pipe),
                             ('freq', df_column_extractor('Frequency'))
                                                ])),
                         ('model', model)
                        ])
    
    ## Pipeline fitting
    pipeline.fit(X_train, y_train)
    
    ## Pipeline Prediction
    pred = pipeline.predict(X_test)
    
    ## Accuracy Scores
    print '\nValidation Accuracy = ',accuracy_score(y_test, pred)
    print '\n',classification_report(y_test, pred)
    print '\nConfusion Matrix:\n', confusion_matrix(y_test, pred)

    #### Following is the code for saving the model
    #### One may comment if model saving not needed

    ## Saving the model
    joblib.dump(pipeline, DIR+'/forest_98_13_Jul.pkl')
    
    ## Loading the model
    ### Validating the Precion, Recall, F1 and Confusion Matrix
    clf = joblib.load(DIR+'/forest_98_13_Jul.pkl')
    
    ## Validation
    new_pred = clf.predict(X_test)
    print '\n\nPickle Accuracy = ',accuracy_score(y_test, new_pred)
    print '\n',classification_report(y_test, new_pred)
    print '\nconfusion_matrix\n',confusion_matrix(y_test, new_pred)
    