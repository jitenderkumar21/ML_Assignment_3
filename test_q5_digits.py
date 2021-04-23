import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork.NeuralNetwork import MLP
from sklearn.preprocessing import MinMaxScaler
from metrics import *
np.random.seed(42)

# loading digits dataset
from sklearn.datasets import load_digits

X,y = load_digits(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
y_new = y.reshape(len(y),1)
object = OneHotEncoder(sparse = False)
y_new = object.fit_transform(y_new)
layers_n = [64, 32]
activ_arr = ["sigmoid", "relu"]
N_L = MLP(X, y_new, layers_n, activ_arr)
N_L.train(X,y_new, 3000)
y_hat = N_L.predict_multiclass(X)
print(accuracy(y_hat, y))



