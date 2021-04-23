import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
np.random.seed(3)

X,y = load_digits(return_X_y=True)
n = 2 # number of components
estimator = PCA(n_components=n)
Xp = estimator.fit_transform(X)

leg = ['0', '1', '2', '3', '4', '5', '7', '8', '9']

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'gold', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        p_x = Xp[:, 0][y == i]
        p_y = Xp[:, 1][y == i]
        plt.scatter(p_x, p_y, c=colors[i])
    
    
plot_pca_scatter()
plt.legend(leg)
plt.title('PCA on digits dataset')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')

plt.show()
