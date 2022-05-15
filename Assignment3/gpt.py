!git clone https://github.com/huggingface/transformers

song=[]

!mkdir content

import os
#os.chdir("transformers")
os.chdir("./transformers/examples/tensorflow/language-modeling")

!pip install -r  requirements.txt


!pip install pyarrow --upgrade

!pip install git+https://github.com/huggingface/transformers

# Commented out IPython magic to ensure Python compatibility.
# %cd /content

from sklearn.model_selection import train_test_split

with open('input.txt', 'r') as data:
  dataset = ["<|title|>" + x.strip() for x in data.readlines()]

train, eval = train_test_split(dataset, train_size=.9, random_state=2020)


with open('train.txt', 'w') as file_handle:
  file_handle.write("<|endoftext|>".join(train))


with open('eval.txt', 'w') as file_handle:
  file_handle.write("<|endoftext|>".join(eval))




!python run_clm.py \
--model_type distilgpt2 \
--model_name_or_path distilgpt2 \
--train_file "train.txt" \
--do_train \
--validation_file "eval.txt" \
--do_eval \
--per_gpu_train_batch_size 1 \
--save_steps -1 \
--num_train_epochs 15 \
--fp16 \
--output_dir="/content/mymodel"

from transformers import TFGPT2LMHeadModel

from transformers import GPT2Tokenizer

model = TFGPT2LMHeadModel.from_pretrained("/content/mymodel/", from_pt=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_ids = tokenizer.encode("I love deep learning", return_tensors='tf')
generated_text_samples = model.generate(
    input_ids, 
    max_length=10000,
    min_length = 500,
    num_return_sequences=20,
    no_repeat_ngram_size=2,
    repetition_penalty=2.5,
    top_p=0.92,
    temperature=.9,
    do_sample=True,
    top_k=125,
    early_stopping=True
)

index=[3,5,6,7,11,13,14,15,17,18,20]



import re


itr=0

for i, b in enumerate(generated_text_samples):
  if(i in index ):
      itr+=1
      song = tokenizer.decode(b, skip_special_tokens=True)
      
      print("=================================Song "+str(itr)+"===============================")
     
      x=song.split('.')
      for s in x:
        if(s.strip()!=''):
          print(s)

      print()