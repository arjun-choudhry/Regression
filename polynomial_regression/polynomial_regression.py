import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

# We dont need the Position column as that directly corelates with the level column
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# We will use all our dataset to train the model

# No feature scaling is required as it is incorporated in the library

# Fitting linear regressor model for comparison
from sklearn.linear_model import LinearRegression
linReg1 = LinearRegression()
linReg1.fit(X,y)

# Fitting polynomial regressor model
from sklearn.preprocessing import PolynomialFeatures
# Now, Creating matrix X_poly from X
polyReg = PolynomialFeatures(degree =3)
X_poly = polyReg.fit_transform(X)

#Now, including X_poly to a linear regression model
linReg2 = LinearRegression()
linReg2.fit(X_poly,y)

# Now, Predicting the values of y using both these models and comparing
y_pred_simpleLinearRegression = linReg1.predict(X)
y_pred_polynomialLinearRegression = linReg2.predict(X_poly)

# Plotting the points for visualization of Simple Linear Regressor
plt.scatter(X,y,color="red")
plt.plot(X,y_pred_simpleLinearRegression,color="blue")
plt.title("Simple Linear Regressor")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Plotting the points for visualization of Polynomial Linear Regressor
#To make the plot continuous
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color="red")
plt.plot(X_grid,linReg2.predict(polyReg.fit_transform(X_grid)),color="blue")
plt.title("Simple Linear Regressor")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()



