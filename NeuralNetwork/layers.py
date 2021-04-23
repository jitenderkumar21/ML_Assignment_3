import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cma
from autograd import grad, elementwise_grad
import autograd.numpy as np

def sigmoid(Z):
    """
    Implementation of sigmoid function
    """
    
    ret = 1/(1+np.exp(-Z))
    cache = Z
    
    return ret, cache

def relu(Z):
    """
    Implementation of relu function
    """
    ret = np.maximum(0,Z)
    cache = Z 
    return ret, cache

def backword_relu(grad, cache):
    """
    Implementation of backword relu function

    """
    dC = np.array(grad,copy=True)
    dC [cache <=0 ] = 0
    return (dC)




def backword_sigmoid(prev_grad,cache):
    """
    Implementation of backword sigmoid function

    """
    Z = cache
    sigmoid = 1/(1+np.exp(-Z))
    dZ = prev_grad*sigmoid*(1-sigmoid)
    return(dZ)


    
def forward_activation(A_prev, W, b, activation="identity"):
    """
    Implement the forward propagation for the layers

    :A_prev:  activations from previous layer (or input data): (size of previous layer, number of examples)
    :W:  Weights matrix(size of current layer, size of previous layer)
    :b:  bias vector (size of the current layer, 1)
    :activation -> activation for current layer

    """
    if activation == "identity":
        Z = np.dot(A_prev, W) + b
        linear_cache = (A_prev, W, b)
        return (Z, linear_cache)

    if activation == "sigmoid":
        
        Z = np.dot(A_prev, W) + b
        linear_cache = (A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache)
        return (A, cache)

    elif activation == "relu":
        Z = np.dot(A_prev, W) + b
        linear_cache = (A_prev, W, b)
        A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)
        return (A, cache)

def backward_activation(prev_grad,cache,activation="identity"):
    """
    Implement the backword propagation for the layers

    """
    # number of features
    

    if(activation=="identity"):
        A_prev , W, bias = cache
        n = A_prev.shape[1] 
        # finding gradients
        dW = (1/n)*np.dot(A_prev.T,prev_grad)
        db = (1/n)*np.sum(prev_grad,axis=0,keepdims=True)
        dA_prev = np.dot(prev_grad,W.T)

        return(dA_prev,dW,db)
    
    elif(activation=="relu"):
        lin_cache , act_cache =cache
        prev_linear_grad = backword_relu(prev_grad,act_cache)

        A_prev , W, bias = lin_cache
        n = A_prev.shape[1] 
        # finding gradients
        dW = (1/n)*np.dot(A_prev.T,prev_linear_grad)
        db = (1/n)*np.sum(prev_linear_grad,axis=0,keepdims=True)
        dA_prev = np.dot(prev_linear_grad,W.T)

        return(dA_prev,dW,db)

    elif(activation=="sigmoid"):
        lin_cache , act_cache =cache
        prev_linear_grad = backword_sigmoid(prev_grad,act_cache)

        A_prev , W, bias = lin_cache
        n = A_prev.shape[1] 
        # finding gradients
        dW = (1/n)*np.dot(A_prev.T,prev_linear_grad)
        db = (1/n)*np.sum(prev_linear_grad,axis=0,keepdims=True)
        dA_prev = np.dot(prev_linear_grad,W.T)

        return(dA_prev,dW,db)
        