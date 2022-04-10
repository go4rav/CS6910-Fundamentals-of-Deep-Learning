
# Overview
The purpose of this assignment was three fold
1. Building and training a CNN model from scratch for iNaturalist image data classification.
2. Fine tune a pretrained model on the iNaturalist dataset.
3. Use a pretrained Object Detection model for a cool application



The link to the wandb report:
https://wandb.ai/go4rav/CS6910%20Assignment2/reports/Assignment-2---VmlldzoxODE3NDI4

## Part A: Building and training a CNN model from scratch for classification:


### Training:
Wandb framework is used to track the loss and accuracy metrics of training and validation. Moreover, bayesian sweeps have been performed for various hyper parameter configurations. 
The sweep configuration and default configurations of hyperparameters are specficied as follows:
```
# configure sweep parameters



sweep_config = {
    
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "val_accuracy",
  "goal": "maximize"
  },
  "parameters": {
                   'base_model': {'values': ['inceptionv3','inceptionresnetv2','resnet50','Xception']},
                   'data_augment': {'values': [False, True]},
                   'batch_norm': {'values': [False, True]}, 
                   "batch_size": { "values": [32, 64] },
                   'dropout_rate': {'values': [0.2, 0.3]},
                   'dense_size': {'values': [128,256]},
                   'epochs': {'values': [5,10]}, 
                }
    
}

sweep_id = wandb.sweep(sweep_config,project='CS6910 Assignment2 partB', entity='go4rav')

# The following is placed within the train() function. 
config_defaults = dict(
                dense_neurons =256 ,
                activation = 'relu',
                num_classes = 10,
                optimizer = 'adam',
                epochs = 5,
                batch_size = 32, 
                img_size = (224,224),
                base_model = "Resnet50"
            ) 



### Testing(Model Flexible):


## Command Line Arguments for Part A:

eg: python file_name.py conv_layer1 conv_layer2 conv_layer3 conv_layer3 conv_layer4 conv_layer5 dropout_rate kernel_size pool_size dense_layer_size data_augmentation batch_norm batch_size num_epochs

## Command Line Aguments for Part B:

eg: python file_name.py base_model_name dense_size data_augment batch_norm batch_size num_epochs dropout_rate



### Visualisation of CNNs:

In order to visualise how the CNNs learn, the following have been implemented through standalone scripts that use the best trained model or any trained keras compatible model for that matter:
1. ```First_layer_filters_visualizations.py``` - filters, the associated feature maps of a specified layer. In our case, it is the first convolutional layer "conv2d"
2. ```guided_backprop.py``` - guided backpropagation on a sample of test images. A guided backpropagation function is implemented that can be generalised to any model that's loaded in keras. 

Both these scripts can be run from the working directory either on the terminal or on the ipython console.


## Part B: Fine tuning a pretrained image classification model.
For this problem, pretrained models such as Xception, ResNet50, InceptionV3, InceptionResnetV2 are used as base models and the user can choose between these models.
The user can also choose to freeze all the layers and make them non trainable and only train the newly added dense layers compatible with the number of classes in the dataset. 
In our case the dense layers were swapped with the output layer having 10 softmax neurons.



### Part C: Real time object detection application using YOLOV5

In this task, YOLOV5s pretrained model was fine tuned one problem:
a) Traffic Analysis

The youtube links are provided in the wandb report mentioned above. 

The python dependencies required for YOLOV5 as given in the [official repository](https://github.com/ultralytics/yolov5)'s requirements.txt can be installed in a virtual environment with python 3.8 and above installed. 


### Evaluations:


## Part A validation accuracy was 41.74%

## Part B validation accuracy was 81.66%
