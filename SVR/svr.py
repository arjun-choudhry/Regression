import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Feature Scaling(required in this part as SVR class doesnt include feature scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scale = sc_X.fit_transform(X)
# X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_scale = sc_y.fit_transform(y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
svrRegressor = SVR(kernel = 'rbf')
svrRegressor.fit(X_scale,y_scale)

y_pred = svrRegressor.predict(X_scale)
y_pred_actual = sc_y.inverse_transform(y_pred)

# Plotting the points for visualization of Regressor result(for higher resolution)
X_grid = np.arange(min(X_scale), max(X_scale), step = 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_scale,y_scale,color="red")
plt.plot(X_grid,svrRegressor.predict(X_grid),color="blue")
plt.title("SVR Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()