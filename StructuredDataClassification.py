# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:14:23 2018

@author: ravul
"""

# Importing required packages
import pandas as pd
from sklearn import preprocessing

# Read data from the churn.csv file
churn = pd.read_csv(r'C:\Users\ravul\Downloads\DKD2e_data_sets\Data sets\churn.csv', sep=',')

# Print the columns
print(churn.columns.values)
'''
# Size of the data set churn
dataSize = churn.shape
print(dataSize)

# Basic stats of the data set
print(churn.describe())

# See a sample data
print(churn.head(3))

# Target variable
churnTarget = churn['Churn?'] 
print(churnTarget)

'''

# Feature Identification and drop the columns from the data set
dropColumns = ['Phone','Churn?'] # Here phone column could be dropped because it doesn't infulence the prediction
churnFeature = churn.drop(dropColumns, axis=1)
print('\nShape of the data set after feature identificaion : ',churnFeature.shape)

# Data preprocessing
## Group the categorical data 
churnFeatureCat = churnFeature.select_dtypes(include=['object'])
print('\nCategorical features in the data set : ',churnFeatureCat.columns.values,
      '\nData types of the categorical features : \n',churnFeatureCat.dtypes)

## Handling categorical data 
### Convert to boolean
boolColumns = ["Int'l Plan", 'VMail Plan']
churnFeature = churnFeature[boolColumns] == 'yes'

### Label Encoding (Technique used to map the non-numerical labels to numerical labels)
print(churnFeature['Area Code'])
labelEncoder = preprocessing.LabelEncoder()
churnFeature['Area Code'] = labelEncoder.fit_transform(churnFeature['Area Code'])
print(churnFeature['Area Code'])

