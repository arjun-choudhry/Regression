# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 03:20:21 2017

@author: arjun
"""

# *************************************************************************************
# AIM: Create a model to predict the salary of employees based on the given data      #
# *************************************************************************************

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderState = LabelEncoder()
X[:,3] = labelEncoderState.fit_transform(X[:,3])

oneHotEncoderState = OneHotEncoder(categorical_features=[3])
X = oneHotEncoderState.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap - Libraries take care of this
X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor = linearRegressor.fit(X_train,y_train)

# Predicting the Test seet result
y_pred = linearRegressor.predict(X_test)