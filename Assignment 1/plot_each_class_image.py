from keras.datasets import fashion_mnist as data
import numpy as np
import pandas as pd
from PIL import Image
!pip install wandb
import wandb
wandb.login()


# Loading dataset
(X_train, Y_train), (X_test, Y_test) = data.load_data()
array=X_train[0,:]    # each image 28x28 pixels
img=Image.fromarray(array)
array.dtype


wandb.init(project="CS6910 Assignment 1", name="q1")  #initialisation


count={}
for i in range(0,10):
	count[i]=0

c=0
index=0


# labels of the Fashion MNIST images
labels = ['top', 'Trouser', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
			   'Ankle boot']

classes=[]

images= []
  

# run the loop while we find 10 unique classes images

while(c<10):
    label=Y_train[index]
    if(count[label]==0):                  # generation
        image=X_train[index,:]
        images.append(image)
        classes.append(labels[label])
        count[label]=1
        c+=1
    index+=1


    
images = [Image.fromarray(image) for image in images]     


wandb.log({"examples": [wandb.Image(image, caption=cap) for image, cap in zip(images,classes)]})  # plotting


wandb.finish()     # finish
