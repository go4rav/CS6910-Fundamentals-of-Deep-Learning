from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
import sys


# importing Keras pre_trained models
from tensorflow.keras.applications import Xception,InceptionV3,InceptionResNetV2,ResNet50




train_dir='./inaturalist_12K/train'
test_dir='./inaturalist_12K/val'



def train():

    # Model Flexible
    # Command line arguments    passed , order => base_model, dense_size, data_augment, batch_norm, batch_size, number of epochs, dropout rate

    base_model = sys.argv[0]
    dense_size = int(sys.argv[1])
    data_augment = bool(sys.argv[2]) # True or False
    batch_norm = bool(sys.argv[3])   # True or False
    batch_size = int(sys.argv[4])
    epochs = int(sys.argv[5])
    dropout_rate = float(sys.argv[6])
    
    input_shape = (256,256,3)
    
    img_height = 256
    img_width =  256






   # fixing image width and height
    img_height = 256
    img_width =  256

    
    # inceptionv3 pretrained model
    if base_model == 'inceptionv3':
        base_model = InceptionV3(include_top=False, weights='imagenet',input_shape=(img_height, img_width,3))
        
    # inceptionresnetv2 pretrained model
    if base_model == 'inceptionresnetv2':
        base_model = InceptionResNetV2(include_top=False, weights='imagenet',input_shape=(img_height, img_width,3))
    
    # resnet50 pretrained model
    if base_model == 'resnet50':
        base_model = ResNet50(include_top=False, weights='imagenet',input_shape=(img_height, img_width,3))
      
    # Xception pretrained model
    if base_model == 'Xception':
        base_model = Xception(include_top=False, weights='imagenet',input_shape=(img_height, img_width,3))  
     
    # Freezing base model layers
    for layers in base_model.layers:
        layers.trainable = False
        
    
    
    with tf.device('/device:GPU:0'):
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
          tf.keras.Input(shape=(img_height, img_width,3,)),
          base_model,
          Flatten(),
          Dense(dense_size,activation='relu'),
        ])

        # batch normalization
        if batch_norm == True:
            model.add(BatchNormalization())
        
        # Adding dropout
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(10 ,activation='softmax'))
            
        # Data augmentation
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
        
        optimiser = tf.keras.optimizers.Adam(learning_rate=0.0004)
        # Model compilation
        model.compile(optimizer=optimiser, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        # Model Fitting
        model.fit(train_data, epochs=epochs, validation_data=valid_data)
        
        # Saving our model
        model.save('./model')
        return model
 





model = train()
 
