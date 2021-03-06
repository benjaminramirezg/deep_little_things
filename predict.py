import pandas as pd
import numpy as np
import tensorflow as tf

import gensim
from gensim.utils import tokenize
from gensim.models import KeyedVectors

import socket
import json

# FILES

embeddings_file = './embeddings.bin'
dataset_file = './train.csv'
output_model_file = './deep_little_things_model'

# EMBEDDINGS

embeddings_model = None #KeyedVectors.load_word2vec_format(embeddings_file, binary=True, unicode_errors='ignore')

# SOCKET CONFIG

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = 'localhost'
port = 1234
buf = 1000000

# NN HYPERPARAMETERS

num_epochs = 50
batch_size=100
input_size=300
output_size=6
hidden_state_size=64
learning_rate=0.01
threshold=0.5

# FUNCTIONS TO PREPARE TEXTS AND LABELS

def _vectorize_token(token):

    if embeddings_model:
        return _vectorize_token_local(token)
    else:
        return _vectorize_token_socket(token)
        

def _vectorize_token_local(token):
    
    try:
        return embeddings_model[token].tolist()
    except:
        return np.zeros(input_size)


def _vectorize_token_socket(token):
    
    try:
        clientsocket.connect((host, port))
        clientsocket.send(token.encode())
        data=clientsocket.recv(buf).decode()
        vector=json.loads(data)
        if not len(vector) == input_size:
            raise Exception("Vector retrieved doesn't match required size " + input_size)
    except:
        vector=np.zeros(input_size)
    return vector

def _vectorize_texts(tokenized_texts,batch_size_):
    max_len=_max_number_of_tokens(tokenized_texts)
    vectorized_texts_by_timestep=[]    
    for timestep_i in range(0, max_len):
        batchitem=[]                    
        for batchitem_i in range(0, batch_size_):
            try:
                token=tokenized_texts[batchitem_i][timestep_i]
                vector=_vectorize_token(token)
                batchitem.append(vector)
            except:
                batchitem.append(np.zeros(input_size).tolist())
        vectorized_texts_by_timestep.append(batchitem)
    return vectorized_texts_by_timestep

def _tokenize_texts(texts):
    tokenized_texts=[]
    for text in texts:
        tokens_list=list(tokenize(text)) # It's a generator
        tokenized_texts.append(tokens_list)
    return tokenized_texts
            

def _max_number_of_tokens(tokenized_texts):
    max_len=0
    for tokenized_text in tokenized_texts:
        if len(tokenized_text) > max_len:
            max_len=len(tokenized_text)
    return max_len

def _get_texts_in_batch(training_set,step,batch_size_):
    try:
        texts_in_batch=training_set.iloc[step*batch_size_:(step+1)*batch_size_,1]
    except:
        texts_in_batch=training_set.iloc[step*batch_size_:,1]
    return texts_in_batch.tolist()

def _get_labels_in_batch(training_set,step,batch_size_):
    try:
        labels_in_batch=training_set.iloc[step*batch_size_:(step+1)*batch_size_,2:]
    except:
        labels_in_batch=training_set.iloc[step*batch_size_:,2:]
    return labels_in_batch.values.tolist()


def _get_input_and_labels_in_batch(training_set,step,batch_size_):
    texts_in_batch=_get_texts_in_batch(training_set,step,batch_size_)
    tokenized_texts=_tokenize_texts(texts_in_batch)
    vectorized_texts=_vectorize_texts(tokenized_texts,batch_size_)
    labels_in_batch=_get_labels_in_batch(training_set,step,batch_size_)
    return vectorized_texts, labels_in_batch


def _shuffle_dataset(dataset):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    return dataset

def split_train_eval_dataset(dataset, evaluation_set_percentage = 20 ):
    dataset=_shuffle_dataset(dataset)
    training_set_n = int(round((((100 - evaluation_set_percentage) * len(dataset)) // 100)))    
    training_set=dataset.head(training_set_n) 
    evaluation_set=dataset.tail(len(dataset) - training_set_n)
    return(training_set,evaluation_set)

##### MAIN

dataset = pd.read_csv(dataset_file)
training_set, evaluation_set = split_train_eval_dataset(dataset)
print('')
print(str(training_set.shape[0]) + ' items in training set')
print(str(evaluation_set.shape[0]) + ' items in evaluation set')
print('')


with tf.Session() as session:
    new_saver=tf.train.import_meta_graph('./deep_little_things_model.meta')
    new_saver.restore(session,tf.train.latest_checkpoint('./'))
    graph=tf.get_default_graph()
    tf_input=graph.get_tensor_by_name("tf_input:0")
    tf_output=graph.get_tensor_by_name("tf_output:0")
    op_predictions=graph.get_tensor_by_name("op_predictions:0")

    for step in range(evaluation_set.shape[0]//batch_size):
        texts_in_batch, labels_in_batch = _get_input_and_labels_in_batch(training_set,step,batch_size)
        print(session.run(op_predictions, {tf_input: texts_in_batch, tf_output: labels_in_batch}))

#####

if not embeddings_model:
    clientsocket.close()
