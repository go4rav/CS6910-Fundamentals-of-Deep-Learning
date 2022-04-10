from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


test_dir='../input/inaturalist-dataset-12K/inaturalist_dataset_12K/val/'


#test data generation
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
  )

test_data = data_gen.flow_from_directory(
      test_dir,
      target_size=(256, 256),
      class_mode='categorical',
      shuffle=True,
      seed=100,
  )

loaded_model = tf.keras.models.load_model("./model")

x_batch, y_batch = next(test_data)

layer_name = 'layer1'
intermediate_layer_model = Model(inputs=loaded_model.input,outputs=loaded_model.layers[1].output)
intermediate_output = intermediate_layer_model.predict(x_batch)
images=[]
for i in range(64) :
    images.append(intermediate_output[0][:,:,i])
figure = plt.figure(figsize=(15, 15.))
grid = ImageGrid(figure, 111,nrows_ncols=(16, 8),axes_pad=0.1)
for axes, img in zip(grid, images):
    axes.imshow(img)
    axes.axis('off')
plt.show()
 
