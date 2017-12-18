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