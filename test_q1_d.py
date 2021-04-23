import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logisticRegression.logisticRegression import LogisticRegression
from metrics import *

np.random.seed(3)
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,n_clusters_per_class=1)
L_R = LogisticRegression(regularization="None")
L_R.fit(X, y)
y_hat = L_R.predict(X)
print(accuracy(y_hat,y))
L_R.plot_decision_boundary(X, y)

