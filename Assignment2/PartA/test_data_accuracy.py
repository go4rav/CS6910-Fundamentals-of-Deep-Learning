from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential


loaded_model = tf.keras.models.load_model("./model")
test_dir='../input/inaturalist-dataset-12K/inaturalist_dataset_12K/val/'


#test data generation
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
  )

test_set = test_datagen.flow_from_directory(
      test_dir,
      target_size=(256, 256),
      class_mode='categorical',
      shuffle=True,
      seed=100,
  )

# model evaluation on test set
loaded_model.evaluate(test_set)
