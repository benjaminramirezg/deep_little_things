import numpy as np
import tensorflow as tf
from PrepareText import DatasetManager
from PrepareText import TokenizationManager
from PrepareText import VectorizationManager
from PrepareText import LabelsManager

# NN HYPERPARAMETERS

num_epochs = 10000
batch_size=20
input_size=300
output_size=2
hidden_states_sizes=[128,128,64]
hidden_state_size=64
learning_rate=0.001
epsilon=0.1
threshold=0.5

# STARTING SESSION AND RESTORING MODEL

session=tf.InteractiveSession()
saver = tf.train.import_meta_graph('./model-many2one.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
tf_input = graph.get_tensor_by_name("tf_input:0")
op_predictions = graph.get_tensor_by_name("op_predictions:0")

# CREATING CLASSES

dataset_manager=DatasetManager(file_name='../data/many2one.csv',file_format='csv', text_position=1, labels_position=0, evaluation_set_percentage = 20)
tokenization_manager=TokenizationManager(tokenizer='gensim')
vectorization_manager=VectorizationManager(vector_size=input_size,embeddings_file='../embeddings/wiki.en.bin', binary_file=True)
labels_manager=LabelsManager(["turn_on_light","set_alarm"])

# PREDICTION

print('', flush=True)
print('==============', flush=True)
print('= PREDICTION =', flush=True)
print('==============', flush=True)

results=[]

for step in range(dataset_manager.evaluation_set.shape[0]//batch_size):

    raw_texts=dataset_manager.get_texts_in_batch('evaluation_set',step,batch_size)
    tokenized_text=tokenization_manager.tokenize_texts(raw_texts)
    vectorized_texts=vectorization_manager.vectorize_texts(tokenized_text,timestep_first=True)
    labels = dataset_manager.get_labels_in_batch('evaluation_set',step,batch_size)
    vectorized_labels=labels_manager.one_hot_vectorize(labels)
    
    prediction = session.run(op_predictions, feed_dict={tf_input: vectorized_texts})
    predicted_labels=labels_manager.get_labels_from_indices(prediction)
    
    for index in range(len(raw_texts)):
        print(raw_texts[index]+'\t'+predicted_labels[index])
