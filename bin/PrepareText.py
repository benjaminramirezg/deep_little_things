import pandas as pd
import numpy as np

import gensim
from gensim.utils import tokenize
from gensim.models import KeyedVectors

import json
import random

class DatasetManager:

    def __init__(self, file_name=None, file_format='csv', text_position=None, labels_position=None, labels_first_position=None, labels_last_position=None, evaluation_set_percentage=0):
        self.file_name=file_name 
        self.file_format=file_format
        self.text_position=text_position
        self.labels_first_position=labels_first_position
        self.labels_position=labels_position        
        self.labels_last_position=labels_last_position
        self.evaluation_set_percentage=evaluation_set_percentage

        if file_format == 'csv':
            self.dataset = pd.read_csv(file_name)
        elif file_format == 'tab':
            self.dataset = pd.read_csv(file_name, sep='\t')
        else:
            raise Exception("Input format non supported")
        
        self.split_train_eval_dataset()


    def shuffle_dataset(self):
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


    def split_train_eval_dataset(self):
        evaluation_set_percentage=self.evaluation_set_percentage
        self.shuffle_dataset()
        training_set_n = int(round((((100 - evaluation_set_percentage) * len(self.dataset)) // 100)))
        self.training_set=self.dataset.head(training_set_n)
        self.evaluation_set=self.dataset.tail(len(self.dataset) - training_set_n)
    

    def get_texts_in_batch(self,set_name,step,batch_size):

        _set=None
        
        if set_name=='training_set':
            _set=self.training_set
        elif set_name=='evaluation_set':
            _set=self.evaluation_set
        else:
            raise Exception('Invalid dataset name')

        try:
            texts_in_batch=_set.iloc[step*batch_size:(step+1)*batch_size,self.text_position]
        except:
            texts_in_batch=_set.iloc[step*batch_size:,self.text_position]

        return texts_in_batch.tolist()


    def get_labels_in_batch(self,set_name,step,batch_size):

        _set=None
        
        if set_name=='training_set':
            _set=self.training_set
        elif set_name=='evaluation_set':
            _set=self.evaluation_set
        else:
            raise Exception('Invalid dataset name')

        if self.labels_position or self.labels_position == 0:
            try:
                labels_in_batch=_set.iloc[step*batch_size:(step+1)*batch_size,self.labels_position]
            except:
                labels_in_batch=_set.iloc[step*batch_size:,self.labels_position]

        else:
            try:
                labels_in_batch=_set.iloc[step*batch_size:(step+1)*batch_size,self.labels_first_position:self.labels_last_position + 1]
            except:
                labels_in_batch=_set.iloc[step*batch_size:,self.labels_first_position:self.labels_last_position + 1]
        return labels_in_batch.values.tolist()


class CoNLLManager():

    def __init__(self, file_name=None, sentence_break=None, evaluation_set_percentage=0):
        self.dataset=[]
        self.training_set=[]
        self.evaluation_set=[]
        self.evaluation_set_percentage=evaluation_set_percentage

        dataset=[]
        current_tokens=[]
        current_labels=[]

        with open(file_name) as f:
            for line in f:
                line=line.strip()
                token, label, remark=line.split('\t')
                current_tokens.append(token)
                current_labels.append(label)
                if remark=='end':
                    dataset.append({'tokens': current_tokens, 'labels': current_labels})
                    current_tokens=[]
                    current_labels=[]
                
        f.closed
        self.dataset=dataset
        self.split_train_eval_dataset()

    def shuffle_dataset(self):
        random.shuffle(self.dataset)

    def split_train_eval_dataset(self):
        evaluation_set_percentage=self.evaluation_set_percentage
        self.shuffle_dataset()
        training_set_n = int(round((((100 - evaluation_set_percentage) * len(self.dataset)) // 100)))
        self.training_set=self.dataset[:training_set_n]
        self.evaluation_set=self.dataset[training_set_n:]

    def get_texts_in_batch(self,set_name,step,batch_size):

        _set=None
        
        if set_name=='training_set':
            _set=self.training_set
        elif set_name=='evaluation_set':
            _set=self.evaluation_set
        else:
            raise Exception('Invalid dataset name')

        try:
            texts_in_batch=_set[step*batch_size:(step+1)*batch_size]
        except:
            texts_in_batch=_set[step*batch_size:]

        return [x['tokens'] for x in texts_in_batch]


    def get_labels_in_batch(self,set_name,step,batch_size):

        _set=None
        
        if set_name=='training_set':
            _set=self.training_set
        elif set_name=='evaluation_set':
            _set=self.evaluation_set
        else:
            raise Exception('Invalid dataset name')

        try:
            labels_in_batch=_set[step*batch_size:(step+1)*batch_size]
        except:
            labels_in_batch=_set.iloc[step*batch_size:]
        return [x['labels'] for x in labels_in_batch]


class Tokenization:

    def __init__(self):
        self.texts=[]
        self.max_len=0
        self.number_of_texts=0
        self.number_of_tokens=0
        self.tokens_dict={}

    def add_tokens_list(self,tokens_list):
        self.texts.append(tokens_list)
        self.max_len=0
        self.number_of_texts=self.number_of_texts + 1
        if len(tokens_list) > self.max_len:
            self.max_len=len(tokens_list)
        for token in tokens_list:

            if token not in self.tokens_dict:
                self.number_of_tokens=self.number_of_tokens + 1
                self.tokens_dict[token]=self.number_of_tokens

class TokenizationManager:

    def __init__(self,tokenizer='gensim'):
        self.tokenizer=tokenizer

    def tokenize_texts(self,texts,tokenization=None):
        if tokenization:
            self.tokenization=tokenization
        else:
            self.tokenization=Tokenization()
        for text in texts:
            tokens_list = self.tokenize_text(text)
            self.tokenization.add_tokens_list(tokens_list)
        return self.tokenization

    def tokenize_text(self,text):
        if self.tokenizer == 'gensim':
            return list(tokenize(text))
        elif self.tokenizer == 'none':
            return text
        else:
            raise Exception("Tokenizer non supported")



class VectorizationManager:

    def __init__(self,vector_size=None,binary_file=True,embeddings_file=None,ids=None):
        self.vector_size=vector_size
        self.embeddings_file=embeddings_file
        self.binary=binary_file
        if embeddings_file:
            self.embeddings=KeyedVectors.load_word2vec_format(self.embeddings_file, binary=binary_file, unicode_errors='ignore')
            self.ids=None
        elif ids:
            self.ids=ids
            self.embeddings=None
        else:
            raise Exception("Embeddings file or ids dicionary needed")
            
                                      

    def vectorize_token(self,token):
        try:
            return self.embeddings[token].tolist()
        except:
            return np.zeros(self.vector_size)
                

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
                if self.embeddings:
                    try:
                        token=tokenization.texts[batchitem_i][timestep_i]
                        vector=self.vectorize_token(token)
                        batchitem.append(vector)
                    except:
                        batchitem.append(np.zeros(self.vector_size).tolist())
                elif self.ids:
                    try:
                        token=tokenization.texts[batchitem_i][timestep_i]
                        batchitem.append(self.ids[token])
                    except:
                        batchitem.append(0)
                else:
                    raise Exception("No embeddings nor ids")

                    
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


class LabelsManager:
    
    def __init__(self,vocabulary):
        self.vector_size=len(vocabulary)
        self.label2vector={}
        self.index2label={}
        for i in range(0,len(vocabulary)):
            word=vocabulary[i]
            vector=np.zeros(len(vocabulary)).tolist()
            vector[i]=1
            self.label2vector[word]=vector
            self.index2label[i]=vocabulary[i]

    def get_labels_from_indices(self,indices):
        labels=[]
        for index in indices:
            label=self.index2label[index]
            labels.append(label)
        return labels

    def get_label_from_index(self,index):
        label=self.index2label[index]
        return label

    def one_hot_vectorize(self,labels):
        vectors=[]
        for label in labels:
            vector=self.label2vector[label]
            vectors.append(vector)
        return vectors

    def one_hot_vectorize_multiple_steps(self,sentences, timestep_first=True):

        if timestep_first:
            tokenization=Tokenization()
            for sentence in sentences:
                tokenization.add_tokens_list(sentence)
            timesteps=tokenization.max_len
            batchsize=tokenization.number_of_texts
            vectors=[]

            for timestep_i in range(0, timesteps):
                batchitem=[]
                for batchitem_i in range(0, batchsize):
                    try:
                        label=tokenization.texts[batchitem_i][timestep_i]
                        vector=self.label2vector[label]
                    except:
                        vector=np.zeros(self.vector_size).tolist()
                    batchitem.append(vector)
                vectors.append(batchitem)
            return vectors

        else:
            vectors=[]
            for sentence in sentences:
                vectors_for_current_sentence=[]
                for label in sentence:
                    vector=self.label2vector[label]
                    vectors_for_current_sentence.append(vector)
                vectors.append(vectors_for_current_sentence)
            return vectors
            
