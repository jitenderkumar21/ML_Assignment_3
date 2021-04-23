import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cma
from autograd import grad, elementwise_grad
import autograd.numpy as np
from NeuralNetwork.layers import *
from sklearn.metrics import mean_squared_error
from scipy.special import softmax,log_softmax

class MLP():
    def _init_parameters(self,layer_n):
        '''
        :layers_n: number of neurons in each layer
        :return para: parameters

        '''
        para = {}
        
        for i in range(1, len(layer_n)):
            # initializing parameters
            para['W' + str(i)]=np.random.normal(loc=0.0,scale = np.sqrt(2/(layer_n[i-1]+layer_n[i])),size = (layer_n[i-1],layer_n[i]))
            para['b'+str(i)] = np.zeros((1, layer_n[i]))
        
        return para

    def __init__(self, inputs,outputs,layers_n, activ_arr, learning_rate = 0.5):
        '''
        :inputs: inputs to the layer
        :activ_arr: activation for each layer

        :return: None
        '''
        self.X = inputs
        self.y = outputs
        self.layers_n = layers_n
        self.activ_arr = activ_arr
        self.learning_rate = learning_rate
        new_matrix = [inputs.shape[1]]
        new_matrix.extend(layers_n)
        new_matrix.append(outputs.shape[1])
        self.para = self._init_parameters(new_matrix)


        pass
    
    def forward(self, inputs):
        
        A = []
        A_curr = inputs
        
        # total number of layers
        n_layers = len(self.para) // 2
        self.caches = []
        for i in range(1,n_layers):

            A_prev = A_curr
            # invoking forward activation 
            A_curr, cache = forward_activation(A_prev,self.para["W" + str(i)], self.para["b"+str(i)], activation = self.activ_arr[i-1])
            A.append(A_curr)
            self.caches.append(cache)
        
        A_curr, cache = forward_activation(A_curr,self.para["W" + str(n_layers)], self.para["b"+str(n_layers)], activation = "identity")
        self.caches.append(cache)
        A.append(A_curr)

        #for digits dataset uncomment this code
        # s = self.softmax(A_curr)
        # A.append(s)

        #Uncomment this for binary classification only
        # s = self.sigmoid_here(A_curr)[0]
        # A.append(s)

        return(A)
    
    def backward(self,grad_loss):
        
        # total number of layers
        n_layers = len(self.para) // 2
        for i in reversed(range(n_layers-1)):
            curr_cache = self.caches[i]
            curr_activation = self.activ_arr[i]
            # invoking backword activation to get the gradients
            grad_loss , dW , db = backward_activation(grad_loss,curr_cache,activation =curr_activation)
            
            # updating parameters using gradients
            self.para["W"+str(i+1)]-= self.learning_rate*dW
            self.para["b"+str(i+1)]-= self.learning_rate*db
    
    def softmax(self,y):
        '''
        softmax implementation using inbuilt function

        '''
        return(softmax(y))

    def sigmoid_here(self, Z):
        """
        Implementation of sigmoid function

        """
        
        ret = 1/(1+np.exp(-Z))
        cache = Z
    
        return ret, cache


    def predict(self, X):
        '''
        Needed for binary classification only
        :return: predicted output
        '''
        AL = self.forward(X)
        prob = AL[-1]
        return (np.where(prob>=0.5,1,0))
    
    def predict_regression(self,input):
        '''
        Needed for Boston dataset
        :return: predicted output
        '''
        AL = self.forward(input)
        prob = AL[-1]
        return(prob)

    def predict_multiclass(self,input):
        '''
        Needed for digits dataset
        :return: predicted output
        '''
        AL = self.forward(input)
        prob = AL[-1]
        vals = np.argmax(prob,axis=1)
        return vals

       


    def train(self,X,y,iterations):
        
        def mse(output,y):
            k = np.mean(np.power(y-output,2))
            return k
            

        for iteration in range(iterations):
            print("Iteration :",iteration,end="\r")
            grad_cost =  elementwise_grad(mse)
            AL = self.forward(X)

            # Uncomment this for digits dataset
            # y_pred = AL[-2]

            # Uncomment this for boston dataset
            y_pred = AL[-1]

            loss = grad_cost(y_pred,y)
            curr_cache = self.caches[-1]

            A_prev , W, bias = curr_cache
            n = A_prev.shape[1] 
            
            # findiing gradients for parameter updates
            dW = (1/n)*np.dot(A_prev.T,loss)
            db = (1/n)*np.sum(loss,axis=0,keepdims=True)
            dA_prev = np.dot(loss,W.T)
    
            # updating parameters using gradients
            self.para["W"+str((len(self.para)//2))]-= self.learning_rate*dW
            self.para["b"+str((len(self.para)//2))]-= self.learning_rate*db
            self.backward(dA_prev)

