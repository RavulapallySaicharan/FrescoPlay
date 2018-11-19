# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:14:23 2018

@author: ravul
"""

# Importing required packages
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import Imputer

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
churnFeature[boolColumns] = churnFeature[boolColumns] == 'yes'

### Label Encoding (Technique used to map the non-numerical labels to numerical labels)
labelEncoder = preprocessing.LabelEncoder()
churnFeature['Area Code'] = labelEncoder.fit_transform(churnFeature['Area Code'])

### One hot encoding (OHE) or Dummy encoding (Technique that maps categorical values onto set of columns that has values 1 or 0 indicating presence of that feature )
print('\nShape of data set before the OHE :', churnFeature.shape,
      '\nNo of Unique states',churnFeature['State'].unique())
churnDumm = pd.get_dummies(churnFeature, columns=['State'], prefix = ['State'])
print('\nShape of the dummies after OHE',churnDumm.shape)
churnMatrix = churnDumm.as_matrix().astype(np.float)
print('\n\n',churnMatrix)

## Missing values
imputer = Imputer(missing_values ='NaN', strategy='mean', axis=0)
churnMatrix = imputer.fit_transform(churnMatrix)
print('\n\n',churnMatrix)

