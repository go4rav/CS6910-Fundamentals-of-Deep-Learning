# -*- coding: utf-8 -*-
"""seq_to_seq_without_attention_flexible.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19CEofwOGuhqnNI780O6ArOZIIyIjDnNt
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import Input,Dense,LSTM,GRU,RNN,SimpleRNN,Softmax,Dropout,Concatenate
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
from attention import AttentionLayer
import pandas as pd

!wget https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
!tar -xvf '/content/dakshina_dataset_v1.0.tar'

def preprocess_data(file_name):
    
    input_texts = []
    target_texts = []
    inputdata=[]
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[: len(lines) - 1]:
        inputdata.append(line)
   
    for line in inputdata:
        target_text,input_text, attestation = line.split("\t")
        
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        
    return(input_texts,target_texts)

input_words, target_words = preprocess_data("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv")


max_encoder_seq_length = max([len(txt) for txt in input_words])
max_decoder_seq_length = max([len(txt) for txt in target_words])

input_characters = set()
target_characters = set()

for input_word in input_words:  
  for char in input_word:
        if char not in input_characters:
            input_characters.add(char)
for target_word in target_words:
    for char in target_word:
        if char not in target_characters:
            target_characters.add(char)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_input_tokens = len(input_characters)
num_target_tokens = len(target_characters)



print(len(input_characters), len(target_characters))
input_char_map = dict([(char, i+1) for i, char in enumerate(input_characters)])
target_char_map = dict([(char, i+1) for i, char in enumerate(target_characters)])
print(len(input_words), len(target_words))
print(input_char_map)


val_input_words, val_target_words = preprocess_data("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv")

test_input_words, test_target_words = preprocess_data("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv")

def one_hot_encoding(input_words, target_words):

    length = len(input_words)
    encoder_input_array = np.zeros(
        (length, max_encoder_seq_length, num_input_tokens+1), dtype="float32"
    )
    decoder_input_array = np.zeros(
        (length, max_decoder_seq_length, num_target_tokens+1), dtype="float32"
    )
    decoder_output_array = np.zeros(
        (length, max_decoder_seq_length, num_target_tokens+1), dtype="float32"
    )


    for i, (input_text, target_text) in enumerate(zip(input_words, target_words)):
        for t, char in enumerate(input_text):
            encoder_input_array[i, t, input_char_map[char]] = 1.0
        
        for t, char in enumerate(target_text):
            
            decoder_input_array[i, t, target_char_map[char]] = 1.0
            if t >=1 :
                
                decoder_output_array[i, t - 1, target_char_map[char]] = 1.0
        
    return(encoder_input_array,decoder_input_array,decoder_output_array)

encoder_input_array, decoder_input_array, decoder_output_array = one_hot_encoding(input_words,target_words)
val_encoder_input_array, val_decoder_input_array, val_decoder_output_array = one_hot_encoding(val_input_words,val_target_words)
test_encoder_input_array, test_decoder_input_array, test_decoder_output_array = one_hot_encoding(test_input_words,test_target_words)

print(decoder_input_array.shape)
encoder_input_array = np.argmax(encoder_input_array, axis=2)
decoder_input_array = np.argmax(decoder_input_array, axis=2)

val_encoder_input_array = np.argmax(val_encoder_input_array, axis=2)
test_encoder_input_array = np.argmax(test_encoder_input_array, axis=2)

val_decoder_input_array = np.argmax(val_decoder_input_array, axis=2)
test_decoder_input_array = np.argmax(test_decoder_input_array, axis=2)

reverse_input_char_map = dict((i, char) for char, i in input_char_map.items())
print(reverse_input_char_map)
reverse_target_char_map = dict((i, char) for char, i in target_char_map.items())
print(reverse_target_char_map)
reverse_target_char_map[0] = "\n"

def define_model(num_cells, cell_type, num_encoder_layers, num_decoder_layers, input_embedding_size, dropout_fraction, beam_size):
    
    encoder_input = keras.Input(shape=(None, ), name="enc_input")
    encoder_embedding = keras.layers.Embedding(num_input_tokens + 1, input_embedding_size, name="enc_embedding", mask_zero=True)(encoder_input)

    
    states = {}
    for i in range(0, num_encoder_layers):
        if cell_type=="LSTM":

            encoder = keras.layers.LSTM(num_cells, return_state=True, return_sequences=True, name="enc_"+str(i+1), dropout=dropout_fraction, recurrent_dropout=dropout_fraction)

            if i==0:
                encoder_outputs, encoder_state_h, encoder_state_c = encoder(encoder_embedding)
            else:
                encoder_outputs, encoder_state_h, encoder_state_c = encoder(encoder_outputs)

            states['encoder_state_h_'+str(i+1)] =  encoder_state_h
            states['encoder_state_c_'+str(i+1)] =  encoder_state_c
              

        if cell_type=="RNN":
  
            encoder = keras.layers.SimpleRNN(num_cells, return_state=True, return_sequences=True, name="enc_"+str(i+1), dropout=dropout_fraction, recurrent_dropout=dropout_fraction)
            
            if i==0:
                whole_sequence_output, rnn_final_state = encoder(encoder_embedding)
            else:
                whole_sequence_output, rnn_final_state = encoder(whole_sequence_output)

            states['rnn_final_state_'+str(i+1)] =  rnn_final_state
            

        if cell_type=="GRU":
            
            encoder = keras.layers.GRU(num_cells, return_state=True, return_sequences=True, name="enc_"+str(i+1), dropout=dropout_fraction, recurrent_dropout=dropout_fraction)
            
            if i==0:
                whole_sequence_output, gru_final_state = encoder(encoder_embedding)
            else:
                whole_sequence_output, gru_final_state = encoder(whole_sequence_output)

            states['gru_final_state_'+str(i+1)] =  gru_final_state
            

   
    decoder_input = keras.Input(shape=(None, ), name="dec_input")
    decoder_embedding = keras.layers.Embedding(num_target_tokens + 1, 64, name="dec_embedding", mask_zero=True)(decoder_input)


    for i in range(0, num_decoder_layers):
        if cell_type=="LSTM":
            decoder_lstm = keras.layers.LSTM(num_cells, return_sequences=True, return_state=True, name="dec_"+str(i+1), dropout=dropout_fraction, recurrent_dropout=dropout_fraction)
            
            if i==0:
                decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_embedding, initial_state = [states['encoder_state_h_'+str(i+1)], states['encoder_state_c_'+str(i+1)]])
            else:
                decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_outputs, initial_state = [states['encoder_state_h_'+str(i+1)],states['encoder_state_c_'+str(i+1)]])
            

        if cell_type=="RNN":
            decoder_rnn = keras.layers.SimpleRNN(num_cells, return_sequences=True, return_state=True, name="dec_"+str(i+1), dropout=dropout_fraction, recurrent_dropout=dropout_fraction)
            if i==0:
                decoder_outputs, rnn_decoder_final_state = decoder_rnn(decoder_embedding, initial_state = states['rnn_final_state_'+str(i+1)])
            else:
                decoder_outputs, rnn_decoder_final_state = decoder_rnn(decoder_outputs, initial_state = states['rnn_final_state_'+str(i+1)])
            
        if cell_type=="GRU":
            decoder_gru = keras.layers.GRU(num_cells, return_sequences=True, return_state=True, name="dec_"+str(i+1), dropout=dropout_fraction, recurrent_dropout=dropout_fraction)
            if i==0:
                decoder_outputs, gru_decoder_final_state = decoder_gru(decoder_embedding, initial_state = states['gru_final_state_'+str(i+1)])
            else:
                decoder_outputs, gru_decoder_final_state = decoder_gru(decoder_outputs, initial_state = states['gru_final_state_'+str(i+1)])
            


    decoder_dense = keras.layers.Dense(num_target_tokens + 1, activation="softmax", name="dec_dense") # Softmax picks one character
    decoder_outputs = decoder_dense(decoder_outputs)


    model = keras.Model([encoder_input, decoder_input], decoder_outputs)

    return model

def inferenceLSTM(model, num_cells):
   
    
    states={}
    enc_states=[]
    enc_inputs = model.input[0]
    dec_inputs = model.input[1]

    
    for layer in model.layers:
        string = layer.name
        i= string[-1]
        if(i.isnumeric() and string[0]=='e'):
          _, enc_h_state, enc_c_state= layer.output
          states['enc_h_state_'+i]=enc_h_state
          states['enc_c_state_'+i]=enc_c_state
          enc_states.append(states['enc_h_state_'+ i])
          enc_states.append(states['enc_c_state_'+ i])
  

    enc_model = keras.Model(enc_inputs, enc_states)

   
    decoders={}
    count=0
    for layer in model.layers:
        if layer.name=="dec_dense":
            dec_dense = layer
        if layer.name == "dec_embedding":
            dec_embedding = layer
        string = layer.name
        i= string[-1]
        if(i.isnumeric() and string[0]=='d'):
          count+=1
          decoders['decoder_'+i]=layer
     

    for i in range(1,count+1):
      input_dec_h_state = keras.Input(shape=(num_cells,))
      input_dec_c_state = keras.Input(shape=(num_cells,))
      states['input_dec_h_state_'+str(i)]=input_dec_h_state
      states['input_dec_c_state_'+str(i)]=input_dec_c_state



    dec_states_inputs=[]
    for i in range(1,count+1):
      states['input_dec_states_'+str(i)]=[]
      states['input_dec_states_'+str(i)].append(states['input_dec_h_state_'+str(i)])
      states['input_dec_states_'+str(i)].append(states['input_dec_c_state_'+str(i)])
      dec_states_inputs= dec_states_inputs+states['input_dec_states_'+str(i)]



    dec_states=[]
    for i in range(1,count+1):
      if(i==1):
        dec_outputs, dec_h_state, dec_c_state = decoders['decoder_'+str(i)](dec_embedding(dec_inputs), states['input_dec_states_'+str(i)])
      else:
        dec_outputs, dec_h_state, dec_c_state = decoders['decoder_'+str(i)](dec_outputs, states['input_dec_states_'+str(i)])
      
      states['dec_h_state_'+str(i)]= dec_h_state
      states['dec_c_state_'+str(i)]= dec_c_state

      dec_states.append(states['dec_h_state_'+str(i)])
      dec_states.append(states['dec_c_state_'+str(i)])


   

   
    dec_outputs = dec_dense(dec_outputs)
   
    dec_model = keras.Model([dec_inputs] + dec_states_inputs, [dec_outputs] + dec_states)

    return enc_model, dec_model

def inferenceOther(model, num_cells):
    
    
    states={}
    enc_states=[]
    enc_inputs = model.input[0]
    dec_inputs = model.input[1]

    
    for layer in model.layers:
        string = layer.name
        i= string[-1]
        if(i.isnumeric() and string[0]=='e'):
          _, enc_state= layer.output
          states['enc_state_'+i]= enc_state
          enc_states.append(states['enc_state_'+ i])
          
  

    
    enc_model = keras.Model(enc_inputs, enc_states)

   

   
    decoders={}
    count=0
    for layer in model.layers:
        if layer.name=="dec_dense":
            dec_dense = layer
        if layer.name == "dec_embedding":
            dec_embedding = layer
        string = layer.name
        i= string[-1]
        if(i.isnumeric() and string[0]=='d'):
          count+=1
          decoders['decoder_'+i]=layer
        
    

    for i in range(1,count+1):
      input_dec_state = keras.Input(shape=(num_cells,))
      states['input_dec_state_'+str(i)]=input_dec_state
      

    

    dec_states_inputs=[]
    for i in range(1,count+1):
      states['input_dec_states_'+str(i)]=[]
      states['input_dec_states_'+str(i)].append(states['input_dec_state_'+str(i)])
      dec_states_inputs= dec_states_inputs+states['input_dec_states_'+str(i)]


    

    dec_states=[]
    for i in range(1,count+1):
      if(i==1):
        dec_outputs, dec_state = decoders['decoder_'+str(i)](dec_embedding(dec_inputs), states['input_dec_states_'+str(i)])
      else:
        dec_outputs, dec_state = decoders['decoder_'+str(i)](dec_outputs, states['input_dec_states_'+str(i)])
      
      states['dec_state_'+str(i)]= dec_state
      

      dec_states.append(states['dec_state_'+str(i)])
 

  
    dec_outputs = dec_dense(dec_outputs)
   
   
    dec_model = keras.Model([dec_inputs] + dec_states_inputs, [dec_outputs] + dec_states)

    
    return enc_model, dec_model

def decode_words(input_words, enc_model, dec_model):
    
    batch_size = input_words.shape[0]
    
    enc_hidden_states = enc_model.predict(input_words)

    target_sequence = np.zeros((batch_size, 1, num_target_tokens+1))
    
    target_sequence[:, 0, target_char_map["\t"]] = 1.0
    target_sequence = np.argmax(target_sequence, axis=2)

    dec_words=[]
    for i in range(batch_size):
      dec_words.append("")

   

    for i in range(max_decoder_seq_length):

        outputs = dec_model.predict([target_sequence] + enc_hidden_states)

        outputs = list(outputs)

        output_tokens = outputs[0]


        sampled_char_indices = np.argmax(output_tokens[:, -1, :], axis=1)

        enc_hidden_states=[]
        
        target_sequence = np.zeros((batch_size, 1, num_target_tokens+1))

        for j, ch_index in enumerate(sampled_char_indices):
            dec_words[j] += reverse_target_char_map[ch_index]
            target_sequence[j, 0, ch_index] = 1.0

        target_sequence = np.argmax(target_sequence, axis=2)

        
        
        for i in range(1,len(outputs)):
          enc_hidden_states.append(outputs[i]) 

    i=0
    for word in dec_words:
      dec_words[i] = word[:word.find("\n")]
      i=i+1
    
    
   
    return dec_words



def train(num_cells, cell_type, num_layers, input_embedding_size, dropout_fraction, beam_size, recurrent_dropout):
   

    model = define_model(num_cells, cell_type, num_layers, num_layers, input_embedding_size, dropout_fraction, beam_size)
    print(model.summary())

   
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    
    history = model.fit(
            [encoder_input_array, decoder_input_array],
            decoder_output_array,
            batch_size = 64,
            epochs = 1,
            verbose = 1,
            validation_data = ([val_encoder_input_array, val_decoder_input_array], val_decoder_output_array)
            )
    
    
    model.save("best_model_without_attention.h5")

    
    

    if cell_type == "LSTM":
        encoder_model, decoder_model = InferenceLSTM(model, num_cells)
    else:
        encoder_model, decoder_model = InferenceOther(model, num_cells)

    

  


    outputs = []
    n = encoder_input_array.shape[0]
    batch_size = 1000
    for i in range(0, n, batch_size):
        
        query = encoder_input_array[i:i+batch_size]
        
        decoded_words = decode_words(query, encoder_model, decoder_model)
        outputs = outputs + decoded_words

   
    ground_truths = [word[1:-1] for word in target_words]
    
    training_inference_accuracy = np.mean(np.array(outputs) == np.array(ground_truths))
    

    outputs = []
    n = val_encoder_input_array.shape[0]
    batch_size = 1000
    for i in range(0, n, batch_size):
       
        query = val_encoder_input_array[i:i+batch_size]
       
        decoded_words = decode_words(query, encoder_model, decoder_model)
        outputs = outputs + decoded_words

  
    ground_truths = [word[1:-1] for word in val_target_words]
    
    validation_inference_accuracy = np.mean(np.array(outputs) == np.array(ground_truths))
   
    

    return model, history

# command line arguments , model flexible

num_cells = int(sys.arg[0])
cell_type = sys.arg[1]
num_layers =int(sys.arg[2])
num_encoder_layers = num_layers
num_decoder_layers = num_layers
input_embedding_size = int(sys.arg[3])
dropout_fraction = int(sys.arg[4])
beam_size = int(sys.arg[5])
recurrent_dropout=int(sys.arg[6])

model,history = train(num_cells, cell_type, num_layers, input_embedding_size, dropout_fraction, beam_size,recurrent_dropout)






