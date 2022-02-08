from keras.datasets import fashion_mnist as data
import numpy as np
import pandas as pd
from PIL import Image
import wandb
wandb.login()


(X_train, Y_train), (X_test, Y_test) = data.load_data()
array=X_train[0,:]    # each image 28x28 pixels
img=Image.fromarray(array)
array.dtype


wandb.init(project="Assignment 1", name="q1")  #initialisation


count={}
for i in range(0,10):
	count[i]=0

c=0
index=0

classes = []
  
while(c<10):
    label=Y_train[index]
    if(count[label]==0):                  # generation
        image=X_train[c,:]
        classes.append(image)
        count[label]=1
        c+=1
    index+=1


    
images = [Image.fromarray(image) for image in classes]     # plotting
wandb.log({"examples": [wandb.Image(image) for image in images]})


wandb.finish()     # finish