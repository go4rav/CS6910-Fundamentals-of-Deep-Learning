
from keras.datasets import fashion_mnist as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image



(X_train, Y_train), (X_test, Y_test) = data.load_data()      # 60000 training examples and 10000 testing examples.


def sigmoid(z):
    a=1/(1+np.exp(-z))
    return(a)


def softmax(z):
    num=np.exp(z)
    den=np.sum(np.exp(z))
    a=num/den
    return(a)

def forwardPropagation(params,layers,X):
    
    a=X
    n=len(layers)-1
    A=[]
    A.append(X)
    for i in range(1,len(layers)-1):
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        z = np.dot(np.transpose(W),a)+b
        a = sigmoid(z)
        A.append(a)
    W=params["W"+str(n)]
    b=params["b"+str(n)]
    z = np.dot(np.transpose(W),a)+b
    a = softmax(z)
    A.append(a)
    return(A)


def sigmoidGradient(a):
    return(a*(1-a))

def backwardPropagate(params, layers,A,alpha,Y):
    l=len(layers)-1
    dz= A[l]-Y
    gradients={}
    while(l>=0):
        dw = np.dot(A[l-1],np.transpose(dz))
        db = dz
        db=db.reshape(db.shape[0],1)
        gradients["dw"+str(l)]=dw
        gradients["db"+str(l)]=db
        if(l>=2):
            da= np.dot(params["W"+str(l)], dz)
            dz = da*sigmoidGradient(A[l-1])
        l=l-1
    return(gradients)
        
def crossEntropyError(a,Y):
    error=-np.sum(Y*np.log(a),axis=0)
    return(error)

     


# m is number of examples
def updateWeights(params,gradients,layers,alpha):
    for i in range(1,len(layers)):
        params["W"+str(i)]-= alpha*gradients["dw"+str(i)]
        params["b"+str(i)]-= alpha*gradients["db"+str(i)]
        
        



#print(X_train[0,:,:])
X_train=X_train.reshape(60000,784)

X_train=np.transpose(X_train)

X_train = X_train/255  # normalised data

Y= Y_train[:]
Y = Y.reshape(X_train.shape[1],1)

Y_train=np.zeros([10,X_train.shape[1]])
for i in range(60000):
  index=Y[i,0]
  Y_train[index,i]=1
Y_train.shape




params={}   # parameters


# input the number of hidden layers and number of hidden units
# X_train layers[0] x m, where m is number of training examples, Y_train is layers[n-1] x m  
# you can handcode the number of layers 
n=3

# you can manually handcode the layers list 
#eg: layers = [5,3,1] here 5 is no.of input layer units, 3 is no.of hidden layer units, 1 is the no.of output layer units

layers=[784,350,10] 

    
for i in range(1,len(layers)):                                   #randomly initialising weights and bias between 0 and 1
    params["W"+str(i)]=np.random.randn(layers[i-1],layers[i])  # weights of layer l are of dimensions layers[l-1]xlayers[l], bias of layer l are of dimension layers[l]x1
    params["b"+str(i)]=np.random.randn(layers[i],1)

   


errors=[]
m=X_train.shape[1]
iters=100
alpha=0.1
while(iters):
  for i in range(m):
    x=X_train[:,i]
    x=x.reshape(x.shape[0],1)
    y=Y_train[:,i]
    y=y.reshape(y.shape[0],1)
    A=forwardPropagation(params, layers,x)
    error= crossEntropyError(A[-1], y)
    errors.append(error)
    gradients=backwardPropagate(params, layers, A, alpha, y)
    updateWeights(params, gradients, layers, alpha)
    print(error)
  iters-=1
   
# A=forwardPropagation(params, layers, X_train)
# print(A[-1])

plt.plot(errors)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")