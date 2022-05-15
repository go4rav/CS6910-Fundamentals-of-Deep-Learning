# Overview
The purpose of this assignment was:
1. Building and training a RNN model from scratch for seq2seq character level Neural machine transliteration.
2. Implement attention based model.


## Dataset:

The dakshina dataset released by google was used for 
In this assignment the Dakshina dataset(https://github.com/google-research-datasets/dakshina) released by Google has been used. This dataset contains pairs of the following form: 
﻿xxx.      yyy﻿
ajanabee अजनबी.
i.e., a word in the native script and its corresponding transliteration in the Latin script (the way we type while chatting with our friends on WhatsApp etc). Given many such (xi,yi)i=1n(x_i, y_i)_{i=1}^n(xi​,yi​)i=1n​ pairs your goal is to train a model y=f^(x)y = \hat{f}(x)y=f^​(x) which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 
These blogs were used as references to understand how to build neural sequence to sequence models: 

https://keras.io/examples/nlp/lstm_seq2seq/
https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a

By default the implemented model uses Hindi as the target language. 

## Building and training a RNN model with and without attention from scratch for sequence to sequence character level neural machine transliteration:



A Bahdanau based attention has been implemented by adapting the code for the bahdanau attention layer class from : https://github.com/thushv89/attention_keras. 
It is advised to setup a virtual environment if running locally using virtualenv/venv and pyenv for python version handling. Or even better, use Conda. But in this assignent I have not used anaconda package manager. 


### Flexible model(CMD arguments):

The sequence to sequence model has been made flexible by passing command line arguments.

```python seq_to_seq_without_attention_flexible.py num_cells cell_type num_layers input_embedding_size dropout_fraction beam_size recurrent_dropout```


### Training:
Wandb framework is used to track the loss and accuracy metrics of training and validation. Moreover, bayesian sweeps have been performed for various hyper parameter configurations. 
The sweep configuration and default configurations of hyperparameters are specficied as follows:
```
sweep_config = {
  'name': 'Attention',
  'method': 'bayes',
  'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
  'parameters': {
      
        'input_embedding':{
            'values' : [32, 64, 128]
        },
        'enc_layers':{
            'values':[1,2,3]
        },
        'dec_layers':{
            'values':[1,2,3]
        },
        'hidden':{
            'values':[64,128,256]
        },
        'cell_type':{
            'values':['GRU', 'LSTM','RNN']
        },
        'dropout':{
            'values':[0.0,0.3]
        },
        'epochs':{
            'values':[5,10,15,20]
        },
        'rec_dropout':{
            'values':[0.0,0.3]
        },
        'beam_size':{
            'values':[1,3]
        }

    }
}

sweep_id = wandb.sweep(sweep_config, project='CS6910 Assignment 3', entity='go4rav')

# The following is placed within the train() function. 
config_defaults = {
        "cell_type": "GRU",
        "latentDim": 256,
        "hidden": 128,
        "optimiser": "rmsprop",
        "numEncoders": 1,
        "numDecoders": 1,
        "dropout": 0.2,
        "epochs": 1,
        "batch_size": 64,
    }
 
```



### Testing

In order to test the best trained model on the test data set, a test script has been written that:
1. Evaluates the test accuracy
2. Saves the predicitons in a csv file
The commands to run the testing script is simply:

testing_and_visualization.ipynb notebook has been included

If one aspires to do further analysis, then it is advised that the test script is run in the ipython console:

```run train.py```

```run test.py```


### Hyperparameter sweeps:

One can find two colab notebooks which are self contained and they can be run on a GPU based runtime session and the results will be logged accordingly in the user entity's wandb account which alone needs to be changed in the notebook before beginning the run. 

