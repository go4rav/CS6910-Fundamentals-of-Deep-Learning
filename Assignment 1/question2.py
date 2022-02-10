from keras.datasets import fashion_mnist as data
import numpy as np
import pandas as pd
from PIL import Image
import wandb
wandb.login()


(X_train, Y_train), (X_test, Y_test) = data.load_data()     



def sigmoid(a):                      # computing sigmoid function of pre-activation units
    h=1/(1+np.exp(-a))
    return(h)

def forwardPropagation(params,layers,X):               # forward pass
    h=X
    A=[]
    A.append(X)
    for i in range(1,len(layers)):
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        a = np.dot(np.transpose(W),h)
        h = sigmoid(a)
        
    return(h)


params={}       

                
# input the number of hidden layers and number of hidden units

n=int(input("Enter number of layers including input layer"))  # X_train layers[0] x m, where m is number of training examples, Y_train is layers[n-1] x m  

layers=[int(i) for i in input("enter each layer's number of input units separated by spaces|").split()]



for i in range(1,len(layers)):									 #randomly initialising weights and bias between 0 and 1
    params["W"+str(i)]=np.random.rand(layers[i-1],layers[i])  # weights of layer l are of dimensions layers[l-1]xlayers[l], bias of layer l are of dimension layers[l]x1
    params["b"+str(i)]=np.random.randn(layers[i],1)

   
h=forwardPropagation(params, layers, X_train)    
print(h)    # output layer