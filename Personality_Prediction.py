# Author: Vishal Mundlye, https://www.linkedin.com/in/vishal-mundlye/
# Developed and tested on:
#   Environment: MacOS Sierra 10.12.06 | 8 Gigs RAM | 125 Gigs hard disk
#   Python: 3.8.5
#   IDE: VSCODE
# Original data source: https://www.kaggle.com/datasnaek/mbti-type
# Training dataset: 1000 records (test_1000.csv)
# Testing dataset: 10 records (test_10.csv)
# Instructions for running the code: Pass the required folder path and 
#   uncomment the respective model related properties and run

# General libraries for working with ML
import io
from csv import reader
import pandas as pd
import numpy as np

# Libraries related to preprocessing, classifiers and models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Libraries related to model evaluation
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# Libraries for Visualizations:
import matplotlib.pyplot as plt
import seaborn as sb

# Libraries related to language/text processors
import random
import os
from tqdm import notebook
import gc
from string import punctuation
import chardet
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup

# Library for saving and loading model
import pickle

   
def cleanText(text, clean_stopwords=True, clean_puntuation=True, clean_numbers=True):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)

    if clean_stopwords == True:
        #Clean Stopwords
        stopword = r"|".join([f"\s{word}\s" for word in stopwords.words("english")])
        f = lambda x : re.sub(stopword, " " , x)
        text = f(text)
    
    if clean_puntuation == True:
        #Clean punctuations, do not remove apostrophe "'' it can make more sense to data
        punctuations = punctuation.replace("'" , "")
        punctuations = f"[{punctuations}]"
        f = lambda x : re.sub(punctuations , "" , text)
        text = f(text)
    
    if clean_numbers == True:
        #Clean Numbers
        f = lambda x : re.sub(r"[0-9]+" , "" , x)
        text = f(text)
    
    return text

def main(filepath):
    # Read data in dataset
    MBTI_full_data = pd.read_csv(filepath + "test_1000.csv")
    #print (MBTI_full_data.head(4))

    # Data Prep: Cleaning data
    MBTI_full_data['clean_posts'] = MBTI_full_data['posts'].apply(cleanText)

    # Preview cleaned data
    #print (MBTI_full_data['clean_posts'].head(4))

    # Data Pre-processing: Creating input feature
    X = MBTI_full_data['clean_posts']

    # Creating target variable
    y = MBTI_full_data['type']

    #Creating Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42) 
    X_train.describe()
    y_train.describe()
    
    # Extracting features from text files:
    # Bag of Words approach used: 
    #   We can save a lot of memory by only storing the non-zero parts of the feature vectors in memory.
    
    #tokenizing the training data
    cv = CountVectorizer()
    X_train = cv.fit_transform(X_train).toarray() #creates a dictionary and the key of words
    #print (X_train.shape)
    #print (cv.vocabulary_.get(u'friend'))  #search for a sample word and its vocabulary token created

    # Not working - need to fix this
    #tfidfconverter = TfidfVectorizer(stop_words="english", max_features=1000, decode_error="ignore")
    #X_train = tfidfconverter.fit_transform(X_train)  
    
    #tokenizing the testing data
    X_test = cv.transform(X_test).toarray()    #transform just creates keys, just need to use transform here not fit_transform
    #X_test = tfidfconverter.transform(X_test).toarray()    #not working

    #Applying classifiers
    # Creating and fitting the Gaussian model
    #gnb_clf = GaussianNB().fit(X_train, y_train)    

    # Creating and fitting a Random Forest model
    #rfc_clf = RandomForestClassifier(n_estimators=1000, random_state=42).fit(X_train, y_train) 

    # Creating and fitting one of the most suitable for word counts, that is the multinomial Naive Bayes
    #mnb_clf = MultinomialNB().fit(X_train, y_train)

    #Linear Support Vector Machine: sgd classification
    sgd_clf= SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None).fit(X_train, y_train)

    #Linear Regression
    #lr_clf = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2').fit(X_train, y_train)

    # Save model to file in the current working directory
    #pkl_filename = "lr_model.pkl"
    pkl_filename = "sgd_model.pkl"
    #pkl_filename = "mnb_model.pkl"
    #pkl_filename = "rfc_model.pkl"
    #pkl_filename = "gnb_model.pkl"
    with open(pkl_filename, 'wb') as file:
       # Save the model
       #pickle.dump(lr_clf, file)
       pickle.dump(sgd_clf, file)
       #pickle.dump(mnb_clf, file)
       #pickle.dump(rfc_clf, file)
        # pickle.dump(gnb_clf, file)

    #Load from file
    with open(pkl_filename, 'rb') as file:
        #Restore pickle model
        #lr_model = pickle.load(file)
        sgd_model = pickle.load(file)
        # mnb_model = pickle.load(file)
        #rfc_model = pickle.load(file)
        # gnb_model = pickle.load(file)

    # Run the model on the testing data set
    # Read data in dataset
    MBTI_10_Test = pd.read_csv(filepath + "test_10.csv")

    ### Data Prep: Cleaning data
    MBTI_10_Test['clean_posts'] = MBTI_10_Test['posts'].apply(cleanText)

    # Data Pre-processing: Creating input feature
    X_test = MBTI_10_Test['clean_posts']

    # Creating target variable
    y_test = MBTI_10_Test['type']

    #vectorize the test features
    X_test = cv.transform(X_test).toarray() #creates a dictionary and the key of words

    # Make predictions
    # gnb_pred = gnb_model.predict(X_test) #88% accuracy
    # rfc_pred = rfc_model.predict(X_test)  #89% accuracy
    #mnb_pred = mnb_model.predict(X_test)  #77% accuracy
    sgd_pred = sgd_model.predict(X_test)  #88% accuracy
    #lr_pred = lr_model.predict(X_test)  #88% accuracy 

    #Evaluating the model:
    #gnb
    # print (classification_report(y_test, gnb_pred))
    # print (confusion_matrix(y_test, gnb_pred))    
    # print(accuracy_score(y_test, gnb_pred))

    #rfc
    # print (classification_report(y_test, rfc_pred))
    # print (confusion_matrix(y_test, rfc_pred))    
    # print(accuracy_score(y_test, rfc_pred))
    
    #mnb
    # print (classification_report(y_test, mnb_pred))
    # print (confusion_matrix(y_test, mnb_pred))    
    # print(accuracy_score(y_test, mnb_pred))

    #sgd
    print (classification_report(y_test, sgd_pred))
    print (confusion_matrix(y_test, sgd_pred))    
    print(accuracy_score(y_test, sgd_pred))

    #Linear Regression:
    # print (classification_report(y_test, lr_pred))
    # print (confusion_matrix(y_test, lr_pred))    
    # print(accuracy_score(y_test, lr_pred))
    

if __name__ == "__main__":
    main("/Users/vish/Documents/Data_Modeling_Analytics_Mining/MBTI_Project/")


