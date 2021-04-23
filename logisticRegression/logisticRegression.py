import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cma
from autograd import grad, elementwise_grad
import autograd.numpy as np
from sklearn.preprocessing import OneHotEncoder

class LogisticRegression():
    def __init__(self, learning_rate=0.1, max_iter=100, regularization='None', lamda = 0.1):
        '''
        :max_iter:  number of iterations
        :regularization:  None, L1 or L2
        :lamda: regularization parameter
        :learning_rate: learning rate

        :return None
        '''
        self.learning_rate  = learning_rate
        self.max_iter       = max_iter
        self.regularization = regularization
        self.lamda          = lamda
        
        self.coef_ = None # to store the learned parameters

        pass


    def fit(self, X, y):
        '''
        :param X: data with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: data corresponding to output (shape: (n_samples,))

        :return None
        '''
        thetas = np.zeros(X.shape[1]+1)
        # adding a column of ones to X 
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        n = X.shape[0]
        for i in range(self.max_iter):
            y_hat = self.sigmoid(np.dot(X,thetas))
            error=y_hat - y

            # checking if regularzation is none
            if self.regularization == 'None':
                # calculating gradient
                gradient = self.learning_rate * (np.dot(X.transpose(), error))
            thetas = thetas - gradient/n

        # updating parameters
        self.coef_ = thetas 


    def fit_autograd(self, X, y):
        '''
        :param X: data with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: data corresponding to output (shape: (n_samples,))

        :return None
        '''
        
        # cost function for regularization = None
        def cost(thetas, x, y):
            y_hat = self.sigmoid(np.dot(x,thetas))
            k = (-1*(y*(np.log(y_hat)) + (1-y)*(np.log(1-y_hat)))).mean(axis=None)
            return k
        
        # cost function for regularization = L1
        def cost_L1(thetas, x, y):
            y_hat = self.sigmoid(np.dot(x,thetas))
            k = -1*(y*(np.log(y_hat)) + (1-y)*(np.log(1-y_hat)))
            extra_term = np.sum(k)
            ans = extra_term + self.lamda*(abs(thetas))
            return ans

        # cost function for regularization = L2
        def cost_L2(thetas, x, y):
            y_hat = self.sigmoid(np.dot(x,thetas))
            k = -1*(y*(np.log(y_hat)) + (1-y)*(np.log(1-y_hat)))
            extra_term = np.sum(k)
            ans = extra_term + self.lamda*(np.dot(thetas.T,thetas))
            return ans

        # autograd function for regularization = None
        def fit_autograd1(X,y):

            grad_cost= grad(cost)
            
            thetas = np.zeros(X.shape[1]+1)
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

            for i in range(self.max_iter):
                # calculating gradient using autograd
                gradient = self.learning_rate * grad_cost(thetas,X,y)
                thetas = thetas - gradient
            
            # updating parameters
            self.coef_ = thetas 

        # cost function for regularization = L1
        def fit_autograd2(X,y):

            grad_cost_L1= elementwise_grad(cost_L1)
            
            thetas = np.zeros(X.shape[1]+1)
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
            n_rows = X.shape[0]

            for i in range(self.max_iter):
                # calculating gradient using autograd
                gradient = (self.learning_rate * grad_cost_L1(thetas,X,y))/n_rows
                thetas = thetas - gradient

            # updating parameters
            self.coef_ = thetas

        # cost function for regularization = L2    
        def fit_autograd3(X,y):
            grad_cost_L2= elementwise_grad(cost_L2)
            
            thetas = np.zeros(X.shape[1]+1)
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
            n_rows = X.shape[0]
            for i in range(self.max_iter):
                # calculating gradient using autograd
                gradient = (self.learning_rate * grad_cost_L2(thetas,X,y))/n_rows
                thetas = thetas - gradient

            # updating parameters
            self.coef_ = thetas  
        
        if self.regularization == 'None':
            fit_autograd1(X,y)
        elif self.regularization == "L1":
            fit_autograd2(X,y)
        elif self.regularization == "L2":
            fit_autograd3(X,y)

    def fit_multi(self, X, y, K):
        '''
        :param X: data with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: data corresponding to output (shape: (n_samples,))
        :K: number of classes

        :return None
        '''

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y = y.reshape(len(y),1)
        object = OneHotEncoder(sparse = False)
        y = object.fit_transform(y)
        thetas = np.zeros([X.shape[1], y.shape[1]])
        n = X.shape[0]
        k = y.shape[1]
        for i in range(self.max_iter):
            total = 0
            for j in range(k):
                total = total + np.exp(np.dot(X,thetas[:,j]))
            for j in range(k):
                s = np.exp(np.dot(X,thetas[:,j]))/total
                e = s - y[:,j]
                thetas[:,j] = thetas[:,j] - ((self.learning_rate * np.dot(e,X))/n)
        
        # updating parameters
        self.coef_ = thetas

    def fit_autograd_multi(self, X, y, K):
        '''
        :param X: data with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: data corresponding to output (shape: (n_samples,))

        :return None
        '''
        
        def cost(thetas, x, y):
            total  = np.exp(np.dot(X,thetas)).sum(axis = -1, keepdims = True)
            log_s  = -np.dot(X,thetas) + np.log(total)
            k = (y*log_s)
            return k
        
        grad_cost = elementwise_grad(cost)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y = y.reshape(len(y),1)
        object = OneHotEncoder(sparse = False)
        y = object.fit_transform(y)
        thetas = np.zeros([X.shape[1], y.shape[1]])
        n = X.shape[0]
        k = y.shape[1]
        for i in range(self.max_iter):
            # calculating gradient using autograd
            gradient = (self.learning_rate * grad_cost(thetas,X,y))/n
            thetas = thetas - gradient
        
        # updating parameters
        self.coef_ = thetas








    def sigmoid(self,z):
        
        return 1/(1+np.exp(-z))
    
    def predict(self, X):
        prob  = self.sigmoid((np.dot(X,self.coef_[1:])+self.coef_[0]))
        y_hat = np.where(prob>=0.5,1,0)
        return y_hat
    
    def predict_multi(self,X):
        prob  = self.sigmoid((np.dot(X,self.coef_[1:])+self.coef_[0]))
        y_hat = np.argmax(prob, axis = 1)
        return y_hat

    
    def plot_decision_boundary(self,X, y):
        cMap = cma.ListedColormap(["#6b76e8", "#c775d1"])
        # cMapa = cma.ListedColormap(["#c775d1", "#6b76e8""#6b76e8", "#c775d1"])

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02  # step size in the mesh

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.predict(np.column_stack((xx.ravel(), yy.ravel())))
        Z = Z.reshape(xx.shape)
        # print(Z)
        plt.figure(1, figsize=(8, 6), frameon=True)
        plt.axis('off')
        plt.pcolormesh(xx, yy, Z, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=y, marker = "o", edgecolors='k', cmap=cMap)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()    