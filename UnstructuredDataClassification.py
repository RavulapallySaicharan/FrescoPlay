# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:08:22 2018

@author: ravul
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

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

print('\nData Frame after Tokenizing :\n', Messages[0:2])    