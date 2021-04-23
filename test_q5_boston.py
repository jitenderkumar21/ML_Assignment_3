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

def split_dataset(X,y,n):
    '''
        Spliting dataset into n folds
    '''
    X_splits = []
    y_splits = []
    t = 0 
    for i in range(n):
        x_splits = []
        yy_splits = []
        l = int(len(X)/n)
        for j in range(t,t+l):
            yy_splits.append(y[j])
            x_splits.append(X[j])
        t=t+l
        y_splits.append(yy_splits)
        X_splits.append(x_splits)
    
    y_splits = np.array(y_splits)
    X_splits = np.array(X_splits)
    return(X_splits, y_splits)

# load the digits dataset
X,y = load_boston(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
y = y.reshape(len(y),1)

X_splits, y_splits = split_dataset(X,y,3)
l = len(X_splits)
flag = 0
avg_mae = 0
avg_rmse = 0
min_mae = np.Inf
min_rmse = np.Inf

for i in range(l):
    x_train = []
    y_train = []
    x_test = X_splits[flag]
    y_test = y_splits[flag]
    for t in range(l):
        if(flag!=t):
            y_train.append(y_splits[t])
            x_train.append(X_splits[t])
    
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    accuracy_d = {}
    y_train = y_train.reshape((y_train.shape[0]*y_train.shape[1],))
    x_train = x_train.reshape((x_train.shape[0]*x_train.shape[1], x_train.shape[2]))

    layers_n = [50, 30]
    activ_arr = ["sigmoid", "relu"]
    y_train = y_train.reshape(len(y_train), 1)
    N_L = MLP(x_train, y_train, layers_n, activ_arr)
    N_L.train(x_train,y_train, 3000)
    y_hat = N_L.predict_regression(x_test)

    mae1 = mae(y_hat,y_test)
    rmse1 = rmse(y_hat,y_test)


    if mae1 < min_mae:
        min_mae = mae1
    
    if rmse1 < min_rmse:
        min_rmse = rmse1
    
    avg_mae = avg_mae + mae1
    avg_rmse = avg_rmse + rmse1
    
    flag = flag +1
    
avg_mae = avg_mae/3
avg_rmse = avg_rmse/3
print("Average mae ", avg_mae)
print("Minimum MAE:", min_mae)
print("Average rmse ", avg_rmse)
print("Minimum RMSE:", min_rmse)



