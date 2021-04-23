import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *

from sklearn.datasets import load_digits
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

# load the digits dataset
X,y = load_digits(return_X_y=True)
X_splits, y_splits = split_dataset(X,y,4)
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



    L_R = LogisticRegression(max_iter = 100, regularization="None")
    L_R.fit_autograd_multi(x_train, y_train, 10)
    y_hat = L_R.predict_multi(x_test)
    acc = accuracy(y_hat,y_test)
    if acc > max_acc:
        max_acc = acc
    
    avg_accuracy =avg_accuracy + acc
    
    flag = flag +1
    
avg_accuracy = avg_accuracy/4
print("Average accuracy ", avg_accuracy)
print("maximum accuracy ", max_acc)


# finding confusion matrix for dataset
X,y = load_digits(return_X_y=True)
L_R = LogisticRegression(max_iter = 100, regularization="None")
L_R.fit_autograd_multi(X, y, 10)
y_hat = L_R.predict_multi(X)
con_matrix  = confusion_matrix(y, y_hat)
print(con_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=con_matrix)
disp.plot() 
plt.show()




