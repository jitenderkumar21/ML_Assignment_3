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
X,y = load_digits(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

X_splits, y_splits = split_dataset(X,y,3)
l = len(X_splits)
flag = 0
avg_accuracy = 0
max_acc = 0
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


    y_new = y_train.reshape(len(y_train),1)
    object = OneHotEncoder(sparse = False)
    y_new = object.fit_transform(y_new)
    layers_n = [64, 32]
    activ_arr = ["sigmoid", "relu"]
    N_L = MLP(X, y_new, layers_n, activ_arr)
    N_L.train(x_train,y_new, 3000)
    y_hat = N_L.predict_multiclass(x_test)

    acc = accuracy(y_hat,y_test)
    if acc > max_acc:
        max_acc = acc
    
    avg_accuracy =avg_accuracy + acc
    
    flag = flag +1
    
avg_accuracy = avg_accuracy/3
print("Average accuracy ", avg_accuracy)
print("maximum accuracy ", max_acc)











