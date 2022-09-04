import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# import tensorflow_hub as hub
import tensorflow as tf
# import bert_tokenization as tokenization
import tensorflow.keras.backend as K
from tensorflow import keras 

import os
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import *

import seaborn as sns
import string
import re    #for regex
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import keras
def load_model(model_file_path):
    model = keras.models.load_model(model_file_path)
    return model

#load tokenizer
from transformers import BertTokenizer
def load_tokenizer():
    MODEL_TYPE = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
    return tokenizer

model=load_model("./")
tokenizer=load_tokenizer()

def return_id(str1, str2, truncation_strategy, length):
    training_sample_count = 1000 # 4000
    test_count = 1000
    MAX_SENTENCE_LENGTH = 20
    MAX_SENTENCES = 5
    MAX_LENGTH = 100

    inputs = tokenizer.encode_plus(str1, str2,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy)

    input_ids =  inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]
    padding_length = length - len(input_ids)
    padding_id = tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, tokenizer):
    training_sample_count = 1000 # 4000
    test_count = 1000
    MAX_SENTENCE_LENGTH = 20
    MAX_SENTENCES = 5
    MAX_LENGTH = 100
    model_input = []
    for xx in range((MAX_SENTENCES*3)+3):
        model_input.append([])
    
    for _, row in tqdm(df[columns].iterrows()):
        i = 0
        
        # sent
        sentences = sent_tokenize(row.text)
        for xx in range(MAX_SENTENCES):
            s = sentences[xx] if xx<len(sentences) else ''
            ids_q, masks_q, segments_q = return_id(s, None, 'longest_first', MAX_SENTENCE_LENGTH)
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1
        
        # full row
        ids_q, masks_q, segments_q = return_id(row.text, None, 'longest_first', MAX_LENGTH)
        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)
        
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)
        
    print(model_input[0].shape)
    return model_input


def input_for_humor_detection(input_str: list,model,tokenizer):
    
    
    input_df=pd.DataFrame(data=input_str,columns=['text'])
    pred_input = compute_input_arrays(input_df, ['text'], tokenizer)
    pred_input = model.predict(pred_input)
    for split in np.arange(0.1, 0.99, 0.1).tolist():
        input_df['pred_bi'] = (pred_input > split)

    #print_evaluation_metrics(df_sub['humor'], df_sub['pred_bi'], '', False, 'SPLIT on '+str(split))

    #input_df.to_csv('sub3.csv', index=False)
    print(input_df.head())
    if input_df['pred_bi'][0]==True:
        print("Hahah you are funny")
    else:
        print("you are not funny")

print(input_for_humor_detection(["All good"],model,tokenizer))



