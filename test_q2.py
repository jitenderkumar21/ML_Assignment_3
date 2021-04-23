import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer         
from metrics import *

from sklearn.datasets import load_breast_cancer
np.random.seed(3)
X,y = load_breast_cancer(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
L_R = LogisticRegression(max_iter = 100, regularization="None")
L_R.fit_autograd(X, y)
y_hat = L_R.predict(X)
acc = accuracy(y_hat,y)
print(acc)
