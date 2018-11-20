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
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedShuffleSplit


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

# Target Identification
churnTarget = churn['Churn?']

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
      '\nNo of Unique states',churnFeature['State'].unique().shape)
churnDumm = pd.get_dummies(churnFeature, columns=['State'], prefix = ['State'])
print('\nShape of the dummies after OHE',churnDumm.shape)
churnMatrix = churnDumm.as_matrix().astype(np.float)
print('\nShape of chrunMatrix :',churnMatrix.shape)

## Missing values
imputer = Imputer(missing_values ='NaN', strategy='mean', axis=0)
churnMatrix = imputer.fit_transform(churnMatrix)
print('\nShape of chrunMatrix after imputer :',churnMatrix.shape)

## Standardization is a technique for re-scaling variables to mean zero and standard deviation to one
scaler = StandardScaler()
churnMatrix = scaler.fit_transform(churnMatrix)
print('\nShape of chrunMatrix after standardization :',churnMatrix.shape)

# Spliting the data into Train(90%) and Test(10%) data sets
seed = 7 # To generate same sequence of random numbers
trainData , testData , trainLabel, testLabel = train_test_split(churnMatrix,churnTarget, test_size = 0.1, random_state = seed)
print('\nShape of the train data :',trainData.shape,
      '\nShape of the train labels :',trainLabel.shape,
      '\nShape of the test data :',testData.shape,
      '\nShape of the test label :',testLabel.shape)


# Classification Algorithms
## Decision Tree Classifier
DTClassifier = DecisionTreeClassifier(random_state = seed)
DTClassifier = DTClassifier.fit(trainData, trainLabel) # Training a Decision Tree model
churnPredicted = DTClassifier.predict(testData) # After fitting model can be used for predicting
score = DTClassifier.score(testData, testLabel)
print('\nDecision Tree Clasifier score is :', score)

## Navie Bayes Clasifier
NBClassifier = GaussianNB()
NBClassifier = NBClassifier.fit(trainData, trainLabel) # Training a Navie Bayes model
score = NBClassifier.score(testData, testLabel)
print('\nNavie Bayes Classifier score is :',score)

## Scochastic Gradient Descent Classifier
SGDclassifier = SGDClassifier()
SGDclassifier = SGDclassifier.fit(trainData, trainLabel) # Training a SGD classifier
score = SGDclassifier.score(testData, testLabel)
print('\nScochastic Gradient Descent Classifier score is :',score)

## SVM Support Vector Machine
SVCClassifier = SVC(kernel='linear', C=0.025, random_state=seed)
SVCClassifier = SVCClassifier.fit(trainData, trainLabel)
score = SVCClassifier.score(testData, testLabel)
print('\nSupport Vector Classifier score is :', score)

## Random Forest Classifier (Ensemble of the Decision tree classifier)
RFClassifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10, random_state = seed)
RFClassifier = RFClassifier.fit(trainData, trainLabel)
score = RFClassifier.score(testData, testLabel)
print('\nRadom Forest Classifier1 score is :', score)
### Model Tuning
RFClassifier = RandomForestClassifier(max_depth=5, n_estimators=15, max_features=60, random_state = seed)
RFClassifier = RFClassifier.fit(trainData, trainLabel)
score = RFClassifier.score(testData, testLabel)
print('\nRadom Forest Classifier2 (tuned) score is :', score)


## Cross Validation

cross_val = StratifiedShuffleSplit(churnTarget,1, test_size=0.1, random_state=seed)

classifiers = [
    ('Decision Tree Classifier',DecisionTreeClassifier()),
    ('Navie Bayes', GaussianNB()),
    ('Scochastic Gradient Descent',SGDClassifier(loss='modified_huber', shuffle=True)),
    ('Support Vector Classifier',SVC(kernel="linear", C=0.025)),
    ('K Nearest Neighbors',KNeighborsClassifier()),
    ('One Vs Rest Classifier', OneVsRestClassifier(LinearSVC())),
    ('Random Forest Classifier',RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10)),
    ('Ada Boost Classifier', AdaBoostClassifier()),
   ]

for clf in classifiers:
    score=0
    for trainIndex, testIndex in cross_val:
        trainDataCV, testDataCV = churnMatrix[trainIndex], churnMatrix[testIndex]
        trainLabelCV, testLabelCV = churnTarget[trainIndex], churnTarget[testIndex]
        clf[1].fit(trainDataCV, trainLabelCV)
        score=score+clf[1].score(testDataCV, testLabelCV)
    print('\n',clf[0],' after Cross Validation :',score)
