# CS6910-Fundamentals-of-Deep-Learning

This is first assignment of the CS6910-Fundamentals-of-Deep-Learning course. The assignment is Classification of Fashion MNIST clothing images using Artificial Neural 
Networks(ANN).   ANN has been  implemented from scratch without using any deep learning frameworks like Keras or Tensorflow.


## Four Core Steps for ANN:

1) Data preprocessing and splitting.
2) Model Training
3) Hyperparameter tuning
4) Evaluation

### 1) Getting Data and preprocessing:

We import Fashion MNIST dataset from Keras.Datasets. 

There are 60000 training images and additional 10000 test images. Each image is of size 28x28. There are 10 classes of images.
We first flatten the input data and then do one hot encoding of each output label.

Helper functions created:

> a) FlattenInput(X_train, X_test)

> b) OneHotEncoding(Y_train,num_train)

Now, we shuffle the input data and split into training data and validation data in the ratio of 9:1. We also normalize the input data.


### 2) Model Training:

After defining the network architecture, there are 4 components in training our model.

 a) Forward Propagation of input data

 b) Computing the error function

 c) Computing gradients through backward propagation

 d) Applying one step of gradient descent algorithm and updating the weights 

We need to repeat the process for fixed number of times.

Helper functions created:

> a) ForwardPropagation()

 feed forwarding our neural network to generate the output
 
> b) CrossEntropyError()

We could use either cross entropy error function or Mean Squared Error Function

> c) BackwardPropagation()

d) Six helper functions for different optimisers

UpdateWeightsSGD()   
UpdateWeightsMomentum()
UpdateWeightsRMS()
UpdateWeightsNesterov()
UpdateWeightsAdam()
UpdateWeightsNAdam()


### 3) HyperParameter Tuning:

We try different models by tuning the hyperparameters using wandb's Random Sweep and get the optimal parameters for our model.

Different hyperparamters are batch_size, number of epochs, weight initialiser, optimiser, learning rate, activation function, weight decay, number of layers and number of hidden neurons of each layer.

Helper Functions: 

InitializeWeights(),  Activation functions like Sigmoid(), Tanh() , Relu() and their gradients SigmoidGradient(), TanhGradient(), ReluGradient()




### 4) Evaluation :

Predict() function for calculating the training, validation and test accuracies.


We get our best model using these hyperparamters:

number of layers=3, number of neurons in each layer=64, number of epochs =10, activation function = ReLU, Optimiser = NAdam , learning rate = 0.001, initialiser = 'xavier', Batch size = 16, weight decay= 0.0005,





