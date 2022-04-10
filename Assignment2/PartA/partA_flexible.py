from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
import sys



train_dir='../input/inaturalist/inaturalist_12K/train'
test_dir='../input/inaturalist/inaturalist_12K/val'



def train():

    # Model Flexible
    # Command line arguments    passed , order => base_model, dense_size, data_augment, batch_norm, batch_size, number of epochs, dropout rate


    conv_layer1= sys.argv[0]
    conv_layer2= sys.argv[1]
    conv_layer3= sys.argv[2]
    conv_layer4= sys.argv[3]
    conv_layer5= sys.argv[4]
    dropout_rate = sys.argv[5]
    kernel_size = sys.argv[6]
    pool_size = sys.argv[7]
    dense_size = sys.argv[8]
    data_augment = sys.argv[9]
    batch_norm = sys.argv[10]
    batch_size = sys.argv[11]
    epochs =  sys.argv[12]
    


    activation = 'relu'
    conv_layers = 5
    input_shape = (256,256,3)


    learning_rate = 0.0004



    filters = [conv_layer1, conv_layer2, conv_layer3, conv_layer4, conv_layer5]

    # if filter organisation is same in each layer
    if filter_org == 1 :
      for i in range(1, conv_layers) :
        filters.append(filters[i - 1])

    # if filter organisation is halves after each layer
    elif filter_org == 0.5 :
      for i in range(1, conv_layers) :
        filters.append(filters[i - 1] / 2)
    
    # # if filter organisation is doubles after each layer
    elif filter_org == 2 :
      for i in range(1, conv_layers) :
        filters.append(filters[i - 1] * 2)

   

    with tf.device('/device:GPU:0'):
      tf.keras.backend.clear_session()
      model = Sequential() 
      for i in range(0,conv_layers): 

          # adding convolution layer
          model.add(Conv2D(filters[i], kernel_size= (kernel_size,kernel_size), input_shape=input_shape, activation=activation))
          # adding max_pooling layer
          model.add(MaxPooling2D(pool_size=(pool_size,pool_size))) 
          # adding batch normalization
          if(batch_norm == True):
            model.add(BatchNormalization())


      # Flattening
      model.add(Flatten()) 

      # Adding a dense layer
      model.add(Dense(dense_size, activation=activation))

      # Adding batch normalsation
      if(batch_norm == True):
            model.add(BatchNormalization())

      # adding dropout 
      model.add(tf.keras.layers.Dropout(rate=dropout_rate))

      # adding output layer
      model.add(Dense(10, activation='softmax'))
      
      model.summary()
      img_height,img_width=(256,256)


      # data augmentation
      if data_augment == True:
        datagen= ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1.0 / 255,
        validation_split=0.1,
        )
      else:
        datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.1,
        )

      train_data = datagen.flow_from_directory(
      train_dir,
      target_size=(img_height, img_width),
      batch_size= batch_size,
      class_mode='categorical',
      shuffle=True,
      subset='training',
      seed=100,
      )

      valid_data = datagen.flow_from_directory(
      train_dir,
      target_size=(img_height, img_width),
      class_mode='categorical',
      shuffle=True,
      subset='validation',
      seed=100,
      )

      optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

      # model compilation
      model.compile(optimizer=optimiser, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

      # model fitting
      model.fit(train_data, epochs=epochs, validation_data=valid_data)

      # model saving
      model.save('./model')

      return model


model = train()
 
