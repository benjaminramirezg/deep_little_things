import numpy as np
import tensorflow as tf
from PrepareText import CoNLLManager
from PrepareText import TokenizationManager
from PrepareText import VectorizationManager
from PrepareText import LabelsManager

num_epochs = 1000
batch_size=2
input_size=300
output_size=5
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

dataset_manager=CoNLLManager(file_name='../data/many2many.txt',sentence_break='end',evaluation_set_percentage = 20)
tokenization_manager=TokenizationManager(tokenizer='none')
vectorization_manager=VectorizationManager(vector_size=input_size,binary_file=True,embeddings_file='../embeddings/wiki.en.bin')
labels_manager=LabelsManager(["ACTION-B","ACTION-I","OBJECT-B","OBJECT-I","NONE"])

# PREDICTION

print('', flush=True)
print('==============', flush=True)
print('= PREDICTION =', flush=True)
print('==============', flush=True)

results=[]

for step in range(len(dataset_manager.evaluation_set)//batch_size):

    raw_texts=dataset_manager.get_texts_in_batch('evaluation_set',step,batch_size)
    tokenized_text=tokenization_manager.tokenize_texts(raw_texts)
    vectorized_texts=vectorization_manager.vectorize_texts(tokenized_text,timestep_first=True)
    labels = dataset_manager.get_labels_in_batch('evaluation_set',step,batch_size)
    vectorized_labels=labels_manager.one_hot_vectorize_multiple_steps(labels,timestep_first=True)

    prediction = session.run(op_predictions, feed_dict={tf_input: vectorized_texts})
    for index in range(len(tokenized_text.texts)):
        for jndex in range(len(tokenized_text.texts[index])):
            try:
                print(tokenized_text.texts[index][jndex]+'\t'+labels_manager.get_label_from_index(prediction[jndex][index]))
            except:
                pass
        print('')
