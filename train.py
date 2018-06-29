import pandas as pd
import numpy as np
import tensorflow as tf

import gensim
from gensim.utils import tokenize
from gensim.models import KeyedVectors

import socket
import json


# EMBEDDINGS

embeddings_model = None # KeyedVectors.load_word2vec_format('embeddings.bin', binary=True, unicode_errors='ignore')

# SOCKET CONFIG

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = 'localhost'
port = 1234
buf = 1000000

# NN HYPERPARAMETERS

num_epochs = 100
batch_size=50
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

def _vectorize_texts(tokenized_texts):
    max_len=_max_number_of_tokens(tokenized_texts)
    vectorized_texts_by_timestep=[]    
    for timestep_i in range(0, max_len):
        batchitem=[]                    
        for batchitem_i in range(0, batch_size):
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

def _get_texts_in_batch(training_set,step):
    try:
        texts_in_batch=training_set.iloc[step*batch_size:(step+1)*batch_size,1]
    except:
        texts_in_batch=training_set.iloc[step*batch_size:,1]
    return texts_in_batch.tolist()

def _get_labels_in_batch(training_set,step):
    try:
        labels_in_batch=training_set.iloc[step*batch_size:(step+1)*batch_size,2:]
    except:
        labels_in_batch=training_set.iloc[step*batch_size:,2:]
    return labels_in_batch.values.tolist()


def _get_input_and_labels_in_batch(training_set,step):
    texts_in_batch=_get_texts_in_batch(training_set,step)
    tokenized_texts=_tokenize_texts(texts_in_batch)
    vectorized_texts=_vectorize_texts(tokenized_texts)
    labels_in_batch=_get_labels_in_batch(training_set,step)
    return vectorized_texts, labels_in_batch

############
### MAIN ###
############

# GRAPH DEFINITION

tf.reset_default_graph()

tf_input=tf.placeholder(tf.float32,shape=[None,None,input_size])
tf_output=tf.placeholder(shape=[None,output_size], dtype=tf.float32)

rnn_cell=tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(rnn_cell, tf_input, initial_state=None, dtype=tf.float32, time_major=True)
output = tf.layers.dense(outputs[-1,:,:], output_size)

loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf_output, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf_output, predictions=tf.to_int32(tf.sigmoid(output) > threshold ))
recall= tf.metrics.recall_at_thresholds(labels=tf_output, predictions=tf.to_int32(tf.sigmoid(output) > threshold ), thresholds=[threshold] )
precision= tf.metrics.precision_at_thresholds(labels=tf_output, predictions=tf.to_int32(tf.sigmoid(output) > threshold ), thresholds=[threshold] )

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# READING INPUT

training_set = pd.read_csv("./train/train.csv")
print('')
print(str(training_set.shape[0]) + ' items in training set')
print('')

# STARTING SESSION

session=tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=4)

# RUNNING SESSION

session.run(init_op)

for epoch in range(num_epochs):
    print('==========')    
    print(str(epoch) + ' epochs')
    print('==========')    
    for step in range(training_set.shape[0]//batch_size):

        texts_in_batch, labels_in_batch = _get_input_and_labels_in_batch(training_set,step) 
        _, loss_ = session.run([train_op,loss], {tf_input: texts_in_batch, tf_output: labels_in_batch})

        if step % 50 == 0:

            accuracy_, _ = session.run(accuracy, feed_dict={tf_input: texts_in_batch, tf_output: labels_in_batch})
            print(str(step) + ' batches (Loss: '+ str(loss_) + ' Accuracy: ' + str(accuracy_) + ')')
            saver.save(session, './deep_little_things_model', global_step=step)


if not embeddings_model:
    clientsocket.close()
