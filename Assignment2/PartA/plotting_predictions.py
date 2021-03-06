from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
import cv2

#test data generation

test_dir='../input/inaturalist-dataset-12K/inaturalist_dataset_12K/val/'

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

# classes
categories=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']

# take sample images

x,y,y_pred=[],[],[]
for category in categories:
    i=0
    path=os.path.join(test_dir,category)
    for img in os.listdir(path):
      if i==3:
        break
      try:
        image = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR)
        x.append(image)
        y.append(category)
        pic = cv2.resize(image, (256,256)) / 255.0
        prediction = model.predict(pic.reshape(1,256,256,3))
        c=prediction.argmax()
        y_pred.append(categories[c])
        i+=1
      except:
        break

#plot sample images with actual and predicted
fig = plt.figure(figsize=(10,20))
rows,columns=10,3
i=1
for k in range(30):

  img=cv2.resize(x[k],(150,150))
  fig.add_subplot(rows,columns,i)
  plt.imshow(img)
  plt.axis('off')
  plt.title('True:'+y[k]+',Predicted:'+y_pred[k],fontdict={'fontsize':10})
  i+=1   
wandb.init(entity='go4rav',project='CS6910 Assignment2 PartB')
wandb.log({'Prediction: ':plt}) 
