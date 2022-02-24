from keras.datasets import fashion_mnist as data
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from optimisers.py import updateWeightsSGD, updateWeightsMomentum, updateWeightsAdam, updateWeightsRMS, updateWeightsNAdam, updateWeightsAdam




def flatten_input(X_train, X_test):
  num_train = X_train.shape[0]
  num_test = X_test.shape[0]
  features = X_train.shape[1]*X_train.shape[2]  # 28x28 = 784
  X_train=X_train.reshape(num_train, features)
  X_test=X_test.reshape(num_test,features)


  X_train=np.transpose(X_train)
  X_test = np.transpose(X_test)

  X_train = X_train/255  # normalised data
  X_test = X_test/255

  return(X_train, X_test)

def OneHotEncoding(Y_train,num_train):
  Y= Y_train[:]
  Y = Y.reshape(num_train,1)

  Y_train=np.zeros([10,num_train])

  for i in range(num_train):
    index=Y[i,0]
    Y_train[index,i]=1
  return(Y_train, Y)





def shuffle_data(X_train, Y_train):
  m=X_train.shape[1]
  permutation = list(np.random.permutation(m))
  X_train = X_train[:, permutation]
  Y_train = Y_train[:, permutation]
  return(X_train,Y_train)





def get_mini_batches(X_train, Y_train,mini_batch_size):
    m=X_train.shape[1]  
    num_batches = m//mini_batch_size
    X_mini_batches = []
    Y_mini_batches = []

    X_train, Y_train = shuffle_data(X_train, Y_train)

    for i in range(num_batches):
      x=X_train[:,i*mini_batch_size:(i+1)*(mini_batch_size)]
      y=Y_train[:,i*mini_batch_size:(i+1)*(mini_batch_size)]
      X_mini_batches.append(x)
      Y_mini_batches.append(y)

    if m%mini_batch_size!=0:
      index = num_batches*mini_batch_size
      x=X_train[:,index:index+m%mini_batch_size]
      y=Y_train[:,index:index+m%mini_batch_size]
      X_mini_batches.append(x)
      Y_mini_batches.append(y)
    
    
    return(X_mini_batches, Y_mini_batches)      


# Activations and their gradients


def relu(Z):

    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    return A

def reluGradient(z):
    dZ = np.zeros(z.shape) 
    dZ[z > 0] = 1
    assert (dZ.shape == z.shape)
    return dZ

def sigmoid(z):
    a=1/(1+np.exp(-z))
    return(a)


def sigmoidGradient(z):
    a=sigmoid(z)
    return(a*(1-a))


def tanh(z):
  a=np.tanh(z)
  return(a)

def tanhGradient(z):
  a=tanh(z)
  return(1-a**2)

def softmax(z):
    num=np.exp(z)
    den=np.sum(np.exp(z),axis=0)
    a=num/den
    return(a)



def initialize_weights(layers, params, M, R, initialiser):
  # seed=3
  # np.random.seed(seed)
  for i in range(1,len(layers)):
      if(initialiser=="xavier"):
        params["W"+str(i)]=np.random.randn(layers[i],layers[i-1])*np.sqrt(2 / (layers[i]+layers[i-1]))
      elif(initialiser=="random"):
        params["W"+str(i)]=np.random.randn(layers[i],layers[i-1])*0.01

      params["b"+str(i)]=np.zeros([layers[i],1])
      M["W"+str(i)]=np.zeros([layers[i],layers[i-1]])
      M["b"+str(i)]=np.zeros([layers[i],1])
      R["W"+str(i)]=np.zeros([layers[i],layers[i-1]])
      R["b"+str(i)]=np.zeros([layers[i],1])
  return(params, M, R)


def feedforward(params,layers,X,activation):
    
    a=X
    n=len(layers)-1
    Z=[]
    A=[]
    A.append(X)
    Z.append(X)
    for i in range(1,len(layers)-1):
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        z = np.dot(W,a)+b
        if(activation=="relu"):
          a = relu(z)
        elif(activation=="sigmoid"):
          a = sigmoid(z)
        elif(activation=="tanh"):
          a = tanh(z)
        A.append(a)
        Z.append(z)
    W=params["W"+str(n)]
    b=params["b"+str(n)]
    z = np.dot(W,a)+b
    a = softmax(z)
    Z.append(z)
    A.append(a)
    return(A,Z)



def backwardPropagate(params, layers, Z, A, learning_rate, Y, activation):
    m=Y.shape[1]
    l=len(layers)-1
    dz= A[l]-Y
    gradients={}
    while(l>=0):
        dw = np.dot(dz,np.transpose(A[l-1]))/m
        db = np.sum(dz,axis=1)/m 
        db=db.reshape(db.shape[0],1)
        gradients["dw"+str(l)]=dw
        gradients["db"+str(l)]=db
        if(l>=2):
            da= np.dot(np.transpose(params["W"+str(l)]), dz)
            if(activation=="relu"):
              dz = da*reluGradient(Z[l-1])
            elif(activation=="sigmoid"):
              dz = da*sigmoidGradient(Z[l-1])
            elif(activation=="tanh"):
              dz = da*tanhGradient(Z[l-1])
        l=l-1
    return(gradients)

        
def crossEntropyError(a,Y,params,layers,weight_decay):
    m=a.shape[1]
    error=-(np.sum(np.sum(Y*np.log(a),axis=1),axis=0))
    #error=-np.sum(Y*np.log(a)+(1-Y)*np.log(1-a))/m
    for i in range(1,len(layers)):
      regu_cost = (weight_decay/2)*np.sum(np.sum(np.square(params["W"+str(i)]),axis=1),axis=0)
    error+=regu_cost
    return(error)
     
def squaredError(a,Y,params,layers,weight_decay):
    m=a.shape[1]
    error=np.sum(np.square(a-Y),axis=1)
    for i in range(1,len(layers)):
      regu_cost = (weight_decay/2)*np.sum(np.sum(np.square(params["W"+str(i)]),axis=1),axis=0)
    error+=regu_cost
    return(error)




def predict(params, layers, X_train, X_test, Y_train_orig, Y_test,activation):
  num_train = X_train.shape[1]
  num_test = X_test.shape[1]
  

  A,Z=feedforward(params, layers, X_train,activation)
  pred=A[-1]
  max_index = np.argmax(pred, axis=0)
  count=0
  for i in range(num_train):
      if(Y_train_orig[0,i]==max_index[i]):
          count+=1
  
  print("train accuracy: ",(count/num_train)*100)
  validationaccuracy = (count/num_train)*100

  A,Z=feedforward(params, layers, X_test,activation)
  pred=A[-1]
  max_index = np.argmax(pred, axis=0)
  count=0
  for i in range(num_test):
      if(Y_test[0,i]==max_index[i]):
          count+=1
  print("test accuracy: ",(count/num_test)*100)

  return((count/num_train)*100)
          
      
  
(X_train, Y_train), (X_test, Y_test) = data.load_data()
(X_train, X_test) = flatten_input(X_train,X_test)


num_train= X_train.shape[1]
num_test=X_test.shape[1]


(Y_train,Y_train_orig)= OneHotEncoding(Y_train,num_train)

Y_train_orig= Y_train_orig.reshape(1,num_train)
Y_test= Y_test.reshape(1,num_test)


print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)


def train_model(X_train, Y_train,X_test, Y_test, Y_train_orig):
	
    mini_batch_size=64
    epochs=10
    initialiser = 'xavier'
    optimiser= 'nadam'
    weight_decay=0
    learning_rate=0.001
    hidden_layers=int(input())
    size_hidden_layer=int(input())
    activation= 'relu'	
	
    layers=[X_train.shape[0]]
    for i in range(hidden_layers):
      layers.append(size_hidden_layer)
    layers.append(Y_train.shape[0])
    params={}
    M={}
    R={}
    lookahead_params={}
    errors=[]
    
    params, M, R = initialize_weights(layers, params, M, R,initialiser)
    (X_mini_batches, Y_mini_batches) = get_mini_batches(X_train, Y_train,mini_batch_size)
    errors=[]
    epoch=1
    gamma1=0.9
    gamma2=0.999
    eps=1e-8
    seed=10
    beta=0.9
    m=X_train.shape[1]
    
    t=0

    
    while(epoch<=epochs):
        error=0
        for i in range(len(X_mini_batches)):
          X_train_mini=X_mini_batches[i]
          Y_train_mini=Y_mini_batches[i]
          A,Z=feedforward(params, layers, X_train_mini,activation)
          error+= crossEntropyError(A[-1], Y_train_mini,params,layers,weight_decay)
          #error=squaredError(A[-1],Y_train)
          if(optimiser=="nadam" or optimiser=="nesterov"):
              for i in range(1,len(layers)):
                lookahead_params["W"+str(i)]=params["W"+str(i)]-M["W"+str(i)]
                lookahead_params["b"+str(i)]=params["b"+str(i)]-M["b"+str(i)]
              lookahead_grads=backwardPropagate(lookahead_params, layers,Z, A, learning_rate, Y_train_mini,activation)
              if(optimiser=="nesterov"):
                params, M = updateWeightsNesterov(num_train, params, lookahead_grads, M, layers, learning_rate, weight_decay, beta)
              elif(optimiser=="nadam"):
                t=t+1
                params,M,R = updateWeightsNAdam(num_train, params, lookahead_grads, layers, M, R, learning_rate, weight_decay, gamma1,gamma2,eps,t)

          else:
              gradients=backwardPropagate(params, layers,Z, A, learning_rate, Y_train_mini, activation)
              if(optimiser=="sgd"):
                params = updateWeightsSGD(num_train, params,gradients,layers, learning_rate, weight_decay)
              elif(optimiser=="momentum"):
                params,M = updateWeightsMomentum(num_train, params, gradients, M, layers, learning_rate,weight_decay, beta)
              elif(optimiser=="rmsprop"):
                params, R = updateWeightsRMS(num_train, params, gradients, R, layers, learning_rate, weight_decay, beta ,eps)
              elif(optimiser=="adam"):
                t=t+1
                params, M, R = updateWeightsAdam(num_train,params, gradients, layers, M, R, learning_rate, weight_decay, gamma1, gamma2, eps, t)
        error=error/m
        errors.append(error)
        print(error,epoch)
        epoch+=1
	
	
      
    #A,Z=feedforward(params, layers, X_train)
    #print(A[-1])

    plt.plot(errors)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss value")

    validationaccuracy=predict(params, layers, X_train, X_test, Y_train_orig, Y_test,activation)
	
	
	
	
train_model()





