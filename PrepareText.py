import pandas as pd
import numpy as np

import gensim
from gensim.utils import tokenize
from gensim.models import KeyedVectors

import socket
import json

class DatasetManager:

    def __init__(self, file_name, file_format='csv', text_position=1, labels_first_position=2):
        self.file_name=file_name
        self.file_format=file_format
        self.text_position=text_position
        self.labels_first_position=labels_first_position

        if file_format == 'csv':
            dataset = pd.read_csv(file_name)
            self.dataset=dataset
        else:
            raise Exception("Input format non supported")

    def shuffle_dataset(self):
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        
    def split_train_eval_dataset(self,evaluation_set_percentage = 20):
        self.shuffle_dataset()
        training_set_n = int(round((((100 - evaluation_set_percentage) * len(self.dataset)) // 100)))
        self.training_set=self.dataset.head(training_set_n)
        self.evaluation_set=self.dataset.tail(len(self.dataset) - training_set_n)
    
    def get_datset_texts_in_batch(self,step,batch_size):
        return self.get_texts_in_batch(self.dataset,step,batch_size)

    def get_datset_labels_in_batch(self,step,batch_size):
        return self.get_labels_in_batch(self.dataset,step,batch_size)

    def get_training_set_texts_in_batch(self,step,batch_size):
        return self.get_texts_in_batch(self.training_set,step,batch_size)

    def get_training_set_labels_in_batch(self,step,batch_size):
        return self.get_labels_in_batch(self.training_set,step,batch_size)

    def get_evaluation_set_texts_in_batch(self,step,batch_size):
        return self.get_texts_in_batch(self.evaluation_set,step,batch_size)

    def get_evaluation_set_labels_in_batch(self,step,batch_size):
        return self.get_labels_in_batch(self.evaluation_set,step,batch_size)

    def get_datset_texts(self):
        step=0
        batch_size=len(self.dataset)
        return self.get_texts_in_batch(self.dataset,step,batch_size)

    def get_datset_labels(self):
        step=0
        batch_size=len(self.dataset)
        return self.get_labels_in_batch(self.dataset,step,batch_size)

    def get_training_set_texts(self):
        step=0
        batch_size=len(self.training_set)
        return self.get_texts_in_batch(self.training_set,step,batch_size)

    def get_training_set_labels(self):
        step=0
        batch_size=len(self.training_set)
        return self.get_labels_in_batch(self.training_set,step,batch_size)

    def get_evaluation_set_texts(self):
        step=0
        batch_size=len(self.evaluation_set)
        return self.get_texts_in_batch(self.evaluation_set,step,batch_size)

    def get_evaluation_set_labels(self):
        step=0
        batch_size=len(self.evaluation_set)
        return self.get_labels_in_batch(self.evaluation_set,step,batch_size)
    
    def get_texts_in_batch(self,training_set,step,batch_size):
        try:
            texts_in_batch=training_set.iloc[step*batch_size:(step+1)*batch_size,self.text_position]
        except:
            texts_in_batch=training_set.iloc[step*batch_size:,self.text_position]
        return texts_in_batch.tolist()

    def get_labels_in_batch(self,training_set,step,batch_size):
        try:
            labels_in_batch=training_set.iloc[step*batch_size:(step+1)*batch_size,self.labels_first_position:]
        except:
            labels_in_batch=training_set.iloc[step*batch_size:,self.labels_first_position:]
        return labels_in_batch.values.tolist()

class Tokenization:

    def __init__(self):
        self.texts=[]
        self.max_len=0
        self.number_of_texts=0

    def add_tokens_list(self,tokens_list):
        self.texts.append(tokens_list)
        self.max_len=0
        self.number_of_texts=self.number_of_texts + 1
        if len(tokens_list) > self.max_len:
            self.max_len=len(tokens_list)

class TokenizationManager:

    def __init__(self,tokenizer='gensim'):
        self.tokenizer=tokenizer

    def tokenize_texts(self,texts):
        self.tokenization=Tokenization()        
        for text in texts:
            tokens_list = self.tokenize_text(text)
            self.tokenization.add_tokens_list(tokens_list)
        return self.tokenization

    def tokenize_text(self,text):
        if self.tokenizer == 'gensim':
            return list(tokenize(text))
        else:
            raise Exception("Tokenizer non supported")

class VectorizationManager:

    def __init__(self,vector_size=300):
        self.vector_size=vector_size

    def set_file_as_embeddings_source(self, embeddings_file, binary_file=True):
        self.embeddings_file=embeddings_file
        self.binary=binary_file
        self.embeddings=KeyedVectors.load_word2vec_format(self.embeddings_file, binary=binary_file, unicode_errors='ignore')
                                      
    def set_socket_as_embeddings_source(self, host='localhost', port=1234, buf=1000000):
        self.host=host
        self.port=port
        self.buf=buf
        self.clientsocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def close_socket(self):
        self.clientsocket.close()

    def vectorize_token(self,token):

        if self.embeddings:
            return self.vectorize_token_local(token)
        else:
            return self.vectorize_token_socket(token)

    def vectorize_token_local(self,token):

        try:
            return self.embeddings[token].tolist()
        except:
            return np.zeros(self.vector_size)

    def vectorize_token_socket(self,token):
        try:
            self.clientsocket.connect((self.host, self.port))
            self.clientsocket.send(token.encode())
            data=self.clientsocket.recv(self.buf).decode()
            vector=json.loads(data)
            if not len(vector) == self.vector_size:
                raise Exception("Vector retrieved doesn't match required size " + self.vector_size)
        except:
            vector=np.zeros(self.vector_size)
        return vector

    def vectorize_texts(self,tokenization,timestep_first=True):
        if timestep_first:
            return self.vectorize_texts_timesteps_first(tokenization)
        else:
            return self.vectorize_texts_batches_first(tokenization)

    def vectorize_texts_timesteps_first(self,tokenization):
        timesteps=tokenization.max_len
        batchsize=tokenization.number_of_texts

        vectorized_texts=[]
        for timestep_i in range(0, timesteps):
            batchitem=[]
            for batchitem_i in range(0, batchsize):
                try:
                    token=tokenization.texts[batchitem_i][timestep_i]
                    vector=self.vectorize_token(token)
                    batchitem.append(vector)
                except:
                    batchitem.append(np.zeros(self.vector_size).tolist())
            vectorized_texts.append(batchitem)
        return vectorized_texts

    def vectorize_texts_batches_first(self,tokenization):
        timesteps=tokenization.max_len
        batchsize=tokenization.number_of_texts

        vectorized_texts=[]

        for batchitem_i in range(0, batchsize):
            timestepitem=[]
            for timestep_i in range(0, timesteps):
                try:
                    token=tokenization.texts[batchitem_i][timestep_i]
                    vector=self.vectorize_token(token)
                    timestepitem.append(vector)
                except:
                    timestepitem.append(np.zeros(self.vector_size).tolist())
            vectorized_texts.append(timestepitem)
        return vectorized_texts
