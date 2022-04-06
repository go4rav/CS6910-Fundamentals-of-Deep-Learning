
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential


import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from tensorflow.python.client import device_lib
device_lib.list_local_devices()




root_path="../input/inaturalist/inaturalist_12K/train/"
root_path2="../input/inaturalist/inaturalist_12K/val/"


import cv2
import os
import numpy as np
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if(filename=='79ed25673a8a0ca05cbcda516c34d55e.jpg' or filename=='8499b5c426f9b475ea79e842ef7d397e.jpg'):
          print(filename)
        img = cv2.imread(os.path.join(folder, filename))
        if(filename=='79ed25673a8a0ca05cbcda516c34d55e.jpg' or filename=='8499b5c426f9b475ea79e842ef7d397e.jpg'):
            print(img.shape)
        if img is not None:
            images.append(img)
    return images
  
  

  
  
  
  
import os
lis = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']
for ele in lis:
    tem  = root_path+ele
    print(tem)
    images=load_images_from_folder(tem)
    print(len(images))
    print(images[0].shape)
    
    
dataset_augment = ImageDataGenerator(rescale=1. / 255)
train = dataset_augment.flow_from_directory(root_path,
                                            shuffle=True, target_size=(256, 256), batch_size=32)
validate = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
root_path2, shuffle=False, target_size=(256,256))
    
  
input_shape = (256,256,3)
with tf.device('/device:GPU:0'):
  # tf.keras.backend.clear_session()
  model = Sequential() 
  for i in range(0,5): 
      model.add(Conv2D(32, kernel_size= (3,3), input_shape=input_shape))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2,2))) 
  model.add(Flatten()) 
  # model.add(Dense(1024, activation='relu'))
  # model.add(tf.keras.layers.Dropout(rate=0.2))
  model.add(Dense(10, activation='softmax'))

  optimiser = tf.keras.optimizers.Adam(learning_rate=.0004)
  model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train, steps_per_epoch=len(train), epochs=2, validation_data=validate)
  # model.add(Dense(64, activation=tf.nn.relu)) 
  # # model.add(Dropout(0.2)) 
  # model.add(Dense(10,activation = tf.nn.softmax))  
    
    
