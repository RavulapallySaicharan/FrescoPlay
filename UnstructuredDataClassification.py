# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:08:22 2018

@author: ravul
"""

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


## Reading text file and breaking it down into list of the samples
file = open(r'C:\Users\ravul\Downloads\SMSSpamCollection.txt','r', encoding='utf-8') # encoding used to accept the ascii characters
Data = file.readlines()
Data = [msg.strip('\n').split('\t;') for msg in Data]    
Messages = pd.DataFrame(Data, columns=['label','message'])
print('\nShape of the data :',Messages.shape) # Shape of the data frame
print('\nColumn names of the data :',list(Messages.columns)) # Column names of the data frame
print('\nStastistics of the data :\n',Messages.describe()) # Understand aggregate stastistics of data frame


## Traget Identification 
messageTarget = Messages['label'] # only 2 unique values. So, it should be a binary classification


## Preprocessing
### Tokenization (In NLP tokenization is initial step for preprocessing)
def splitTokens(msg):
    msg = msg.lower() # change everything to lower case
    #msg = unicode(msg, 'utf8') # converts bytes into proper unicode
    wordTokens = word_tokenize(msg)
    return(wordTokens)
    
Messages['tokenized message'] = Messages.apply(lambda row: splitTokens(row['message']), axis=1)
'''print('\nData Frame after Tokenizing :\n', Messages[0:2]) '''   

### Lemmatization (Method to convert the words into its basic/root form removes affixes of the word (ex : rules to rule, swims to swim))
def splitIntoLemmas(msg):
    lemma = []
    lemmantizer = WordNetLemmatizer()
    for wrd in msg:
        a = lemmantizer.lemmatize(wrd)
        lemma.append(a)
    return lemma

Messages['lemmantized words'] = Messages.apply(lambda row: splitIntoLemmas(row['tokenized message']), axis=1)
'''print('\nData Frame after Lemmantization :\n', Messages[0:2])'''

### Stop word removal (Stop words doesn't add any relavance to the classification)
def stopWordRemoval(msg):
    stopWords = set(stopwords.words('english'))
    filteredSentence = []
    filteredSentence = ' '.join([word for word in msg if word not in stopWords])
    return filteredSentence

Messages['preprocessed message'] = Messages.apply(lambda row: stopWordRemoval(row['lemmantized words']), axis=1)
'''print('\nData Frame after stop word removal: \n', Messages[0:2])'''

TrainingData = pd.Series(list(Messages['preprocessed message']))
TrainingLabel = pd.Series(list(Messages['label']))

## Feature Extraction (Convert the text content into the vector form)
### Bag of Words(BOW) most widely used method for generating features in NLP used for calculating the word frequency which can be used as feature for training a classifier
### TDM (Term Document Matrix) is the matrix that contain the frequencies of occurances of terms in collection of documents, rows correspond documents and columns correspond terms

tfVectorizer = CountVectorizer(ngram_range=(1,2),min_df=(1/len(TrainingLabel)),max_df=0.7)
totalDictionaryTDM = tfVectorizer.fit(TrainingData)
messageDataTDM = totalDictionaryTDM.transform(TrainingData)

print(messageDataTDM.shape)

### Term Frequency Inverse Document Frequency (TFIDF), IDF diminishes the weight of most common occuring words and increases the weightage of the rare words

tfIdfVectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=(1/len(TrainingLabel)),max_df=0.7)
totalDictionaryTFIDF = tfIdfVectorizer.fit(TrainingData)
messageDataTFIDF = totalDictionaryTFIDF.transform(TrainingData)

print(messageDataTFIDF.shape)


## Splitting the training and test data
trainData, testData,trainLabels, testLabels = train_test_split(messageDataTDM, TrainingLabel, test_size=0.1)
print('\nTrain data shape :',trainData.size,
      '\nTrain label shape :',trainLabels.shape,
      '\nTest data shape :',testData.shape,
      '\nTest label shape :',testLabels.shape)

## Classification
### Decision Tree Classification
DTclassifier = DecisionTreeClassifier()
DTclassifier = DTclassifier.fit(trainData, trainLabels)
score = DTclassifier.score(testData, testLabels)
print('\nScore of decision tree classifier :',score)

### Stochastic Gradient Descent
SGDclassifier = SGDClassifier()
SGDclassifier = SGDclassifier.fit(trainData, trainLabels)
score = SGDclassifier.score(testData, testLabels)
print('\nScore of stochastic gradient descent :',score)

### Support Vector Machine (SVM)
SVMclassifier = SVC(kernel='linear', C=0.25, random_state = 7)
SVMclassifier = SVMclassifier.fit(trainData, trainLabels)
score = SVMclassifier.score(testData, testLabels)
print('\nScore of the SVM classifier :',score )

### Random Forests
RFclassifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10, random_state=7)
RFclassifier = RFclassifier.fit(trainData, trainLabels)
score = RFclassifier.score(testData, testLabels)
print('\nScore of the Random Forests :',score)
#### Model Tuning
RFclassifier = RandomForestClassifier(max_depth=10, n_estimators=15, max_features=60, random_state=7)
RFclassifier = RFclassifier.fit(trainData, trainLabels)
score = RFclassifier.score(testData, testLabels)
print('\nModel Tuned\nScore of the Random Forests :',score)
