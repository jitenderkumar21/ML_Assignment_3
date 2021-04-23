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


X,y = load_breast_cancer(return_X_y=True)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
# print(X.shape)
# print(y.shape)

#5 fold nested cross validation
X_splits, y_splits = split_dataset(X,y,5)
# print(X_splits.shape)
# print(y_splits.shape)

l = len(X_splits)
# testing values of lamda
values = [0.005, 0.05, 0.5, 5, 50, 500]

flag = 0
op = []
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

    for val in values:
        accuracy1 = []
        flag1 = 0 
        XX_splits, yy_splits = split_dataset(x_train, y_train, 5)
        # print(XX_splits.shape)
        # print(yy_splits.shape)

        for j in range(len(XX_splits)):
            xx_train = []
            yy_train = []
            xx_test = XX_splits[flag1]
            yy_test = yy_splits[flag1]

            for t in range(len(XX_splits)):
                if(flag1!=t):
                    yy_train.append(yy_splits[t])
                    xx_train.append(XX_splits[t])
            yy_train = np.array(yy_train)
            xx_train = np.array(xx_train)
            yy_train = yy_train.reshape((yy_train.shape[0]*yy_train.shape[1],))
            xx_train = xx_train.reshape((xx_train.shape[0]*xx_train.shape[1], xx_train.shape[2]))
            flag1 = flag1 +1
            L_R = LogisticRegression(regularization="L1", lamda = val)
            L_R.fit_autograd(xx_train,yy_train)
            y_hat = L_R.predict(xx_test)
            current_acc = accuracy(y_hat, yy_test)
            # print(current_acc)
            accuracy1.append(current_acc)
        accuracy_d[val] = np.mean(accuracy1)
    flag = flag + 1
    maximum_accr = 0
    print(accuracy_d)
    best_val = values[0]
    for tt in accuracy_d:
        if(maximum_accr<=accuracy_d[tt]):
            maximum_accr = accuracy_d[tt]
            best_val = tt
    op.append(best_val)
    print(maximum_accr)

print(op)











