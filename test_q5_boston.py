import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork.NeuralNetwork import MLP
from sklearn.preprocessing import MinMaxScaler
from metrics import *
np.random.seed(42)

# loading digits dataset
from sklearn.datasets import load_boston

X,y = load_boston(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
y = y.reshape(len(y),1)


layers_n = [50, 30]
activ_arr = ["sigmoid", "relu"]
N_L = MLP(X, y, layers_n, activ_arr)
N_L.train(X,y, 3000)
y_hat = N_L.predict_regression(X)
print(y_hat)
print(y_hat.shape)
print("MAE:",mae(y_hat,y))
print("RMSE:",rmse(y_hat,y))


