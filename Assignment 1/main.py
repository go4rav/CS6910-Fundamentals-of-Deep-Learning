from keras.datasets import fashion_mnist as data
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from optimisers.py import updateWeightsMomentum, updateWeightsAdam, updateWeightsRMS




(X_train, Y_train), (X_test, Y_test) = data.load_data()

num_train = X_train.shape[0]
num_test = X_test.shape[0]
features = X_train.shape[1]*X_train.shape[2]  # 28x28
#print(X_train[0,:,:])
X_train=X_train.reshape(num_train, features)
X_test=X_test.reshape(num_test,features)

X_train=np.transpose(X_train)
X_test=np.transpose(X_test)


X_train = X_train/255  # normalised data
X_test = X_test/255

Y= Y_train[:]
Y = Y.reshape(X_train.shape[1],1)

Y_train=np.zeros([10,X_train.shape[1]])

for i in range(num_train):
  index=Y[i,0]
  Y_train[index,i]=1

print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)


def sigmoid(z):
    a=1/(1+np.exp(-z))
    return(a)

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    return A

def reluGradient(z):
    
    
    dZ = np.zeros(z.shape) # just converting dz to a correct object.
    
    
    dZ[z > 0] = 1
    
    assert (dZ.shape == z.shape)
    
    return dZ


def softmax(z):
    num=np.exp(z)
    den=np.sum(np.exp(z),axis=0)
    a=num/den
    return(a)

def feedforward(params,layers,X):
    
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
        a = relu(z)
        A.append(a)
        Z.append(z)
    W=params["W"+str(n)]
    b=params["b"+str(n)]
    z = np.dot(W,a)+b
    a = sigmoid(z)
    Z.append(z)
    A.append(a)
    return(A,Z)


def sigmoidGradient(z):
    a=sigmoid(z)
    return(a*(1-a))

def backwardPropagate(params, layers,Z,A,alpha,Y):
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
            dz = da*reluGradient(Z[l-1])
        l=l-1
    return(gradients)
        
def crossEntropyError(a,Y):
    m=a.shape[1]
    #error=-(np.sum(np.sum(Y*np.log(a),axis=1),axis=0)/m)
    error=-np.sum(Y*np.log(a)+(1-Y)*np.log(1-a))/m
    return(error)
     
def squaredError(a,Y):
    m=a.shape[1]
    error=np.sum(np.square(a-Y),axis=1)/m
    return(error)

def updateWeights(params,gradients,layers,alpha,optimiser,M,R,gamma1,gamma2,beta,eps,t):
	if(optimiser=="momentum"):
		return(updateWeightsMomentum(params,gradients,layers,alpha,beta,M))
	else if(optimiser=="adam"):
		return(updateWeightsAdam(params,gradients,layers,alpha,gamma1,gamma2,eps,t,M,R))
	else if(optimiser=="RMS"):
		return(updateWeightsRMS(params,gradients,layers,alpha,beta,R))


 

def initialize_weights():
  seed=3
  np.random.seed(seed)
  for i in range(1,len(layers)):
      params["W"+str(i)]=np.random.randn(layers[i],layers[i-1])*np.sqrt(2 / layers[i-1])
      params["b"+str(i)]=np.zeros([layers[i],1])
      M["W"+str(i)]=np.zeros([layers[i],layers[i-1]])
      M["b"+str(i)]=np.zeros([layers[i],1])
      R["W"+str(i)]=np.zeros([layers[i],layers[i-1]])
      R["b"+str(i)]=np.zeros([layers[i],1])
      






# number of layers and each layer number of units are flexible


layers=[784,32,32,32,10]

params={}

M={}
R={}
errors=[]
initialize_weights()


# X_train=np.array([[0,0,1,1],[0,1,0,1]])
# Y_train=np.array([[0,1,1,0]])

m=X_train.shape[1]
batch_size=64
X_mini_batches=[]
Y_mini_batches=[]
num_batches=m//batch_size
for i in range(num_batches):
  x=X_train[:,i*batch_size:(i+1)*(batch_size)]
  y=Y_train[:,i*batch_size:(i+1)*(batch_size)]
  X_mini_batches.append(x)
  Y_mini_batches.append(y)

if m%batch_size!=0:
  index = num_batches*batch_size
  x=X_train[:,index:index+m%batch_size]
  y=Y_train[:,index:index+m%batch_size]
  X_mini_batches.append(x)
  Y_mini_batches.append(y)


errors=[]
gamma1=0.9
gamma2=0.999
eps=1e-8
iters= 10000
alpha=0.001
seed=1000
t=0

# you can use "adam", "momentum", "RMS"
optimiser="adam"      


# mini_batch gradient descent


while(iters):
    error=0
    for i in range(len(X_mini_batches)):
      X_train_mini=X_mini_batches[i]
      Y_train_mini=Y_mini_batches[i]
      A,Z=feedforward(params, layers, X_train_mini)
      error+= crossEntropyError(A[-1], Y_train_mini)
      #error=squaredError(A[-1],Y_train)
      gradients=backwardPropagate(params, layers,Z, A, alpha, Y_train_mini)
      t=t+1
      updateWeights(params,gradients,layers,alpha,optimiser,M,R,gamma1,gamma2,beta,eps,t,optimiser)
    if(iters%1000==0):
      errors.append(error/m)
      print(error/m,iters)
    iters-=1
   
#A,Z=feedforward(params, layers, X_train)
#print(A[-1])

plt.plot(errors)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
