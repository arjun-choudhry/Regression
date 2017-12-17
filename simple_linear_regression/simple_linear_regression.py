# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:18:06 2017

@author: arjun
"""
# *************************************************************************************
# AIM: Create a model to predict the salary of employees based on the given data      #
# *************************************************************************************

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# no categorization required

# Splitting the dataset to training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor = linearRegressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = linearRegressor.predict(X_test)

# Plotting the graphs to visualize the training set results
# The below plot will formulate the scatter plot
plt.scatter(X_train, y_train,color = 'red')
# The below will plot the regressor line. The X values will be the X-train, whereas the y values will be the predicted linear regression model values of the X_train.
plt.plot(X_train, linearRegressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Plotting the results of the test results
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()






