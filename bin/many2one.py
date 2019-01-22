import numpy as np
import tensorflow as tf
from PrepareText import DatasetManager
from PrepareText import Tokenization
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

# CREATING CLASSES

dataset_manager=DatasetManager(file_name='../data/many2one.csv',file_format='csv', text_position=1, labels_position=0, evaluation_set_percentage = 20)
'''
all_training_texts=dataset_manager.get_texts_in_batch('training_set',0,len(dataset_manager.dataset))
all_evaluation_texts=dataset_manager.get_texts_in_batch('evaluation_set',0,len(dataset_manager.dataset))
'''
tokenization_manager=TokenizationManager(tokenizer='gensim')
'''
tokenization=Tokenization()
tokenization_manager.tokenize_texts(all_training_texts,tokenization)
tokenization_manager.tokenize_texts(all_evaluation_texts,tokenization)
ids_dict=tokenization.tokens_dict
number_of_tokens=tokenization.number_of_tokens + 1 # We start in 1 because 0 is unknown. So tne range must be +1
'''
vectorization_manager=VectorizationManager(vector_size=input_size,embeddings_file='../embeddings/wiki.en.bin', binary_file=True)
#vectorization_manager=VectorizationManager(ids=ids_dict)
labels_manager=LabelsManager(["turn_on_light","set_alarm"])

# GRAPH DEFINITION

tf.reset_default_graph()

tf_input=tf.placeholder(shape=[None,None,input_size], dtype=tf.float32, name='tf_input')
#tf_input=tf.placeholder(shape=[None,None], dtype=tf.int32, name='tf_input')
tf_output=tf.placeholder(shape=[None,output_size], dtype=tf.float32, name='tf_output')

#word_embeddings = tf.get_variable("word_embeddings", [number_of_tokens, input_size])
#embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, tf_input)

rnn_cells=tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=n) for n in hidden_states_sizes])
outputs, state = tf.nn.dynamic_rnn(rnn_cells, tf_input, dtype=tf.float32, time_major=True)
#outputs, state = tf.nn.dynamic_rnn(rnn_cells, embedded_word_ids, dtype=tf.float32, time_major=True)
output=tf.layers.dense(outputs[-1,:,:], output_size)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_output, logits=output)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

op_predictions = tf.argmax(tf.nn.softmax(output), axis=1, name='op_predictions')
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_output, axis=1), predictions=op_predictions)

# CREATING SAVER

saver = tf.train.Saver()

# STARTING SESSION

session=tf.InteractiveSession()

# RUNNING SESSION

session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))


# TRAINING

print('', flush=True)
print('============', flush=True)
print('= TRAINING =', flush=True)
print('============', flush=True)


for epoch in range(num_epochs):
    print('', flush=True)
    print('=== ' + str(epoch) + ' epochs ===', flush=True)
    print('', flush=True)    
    for step in range(dataset_manager.training_set.shape[0]//batch_size):

        raw_texts=dataset_manager.get_texts_in_batch('training_set',step,batch_size)
        tokenized_texts=tokenization_manager.tokenize_texts(raw_texts)
        vectorized_texts=vectorization_manager.vectorize_texts(tokenized_texts,timestep_first=True)
        labels = dataset_manager.get_labels_in_batch('training_set',step,batch_size)
        vectorized_labels=labels_manager.one_hot_vectorize(labels)
        _, loss_ = session.run([train_op,loss], {tf_input: vectorized_texts, tf_output: vectorized_labels})

        if step % 1 == 0:
            print(str(step) + ' batches (Loss: '+ str(loss_) + ')', flush=True)

# SAVING MODEL

saver.save(session, './model-many2one')

# EVALUATION

print('', flush=True)
print('==============', flush=True)
print('= EVALUATION =', flush=True)
print('==============', flush=True)

results=[]

for step in range(dataset_manager.evaluation_set.shape[0]//batch_size):

    raw_texts=dataset_manager.get_texts_in_batch('evaluation_set',step,batch_size)
    tokenized_text=tokenization_manager.tokenize_texts(raw_texts)
    vectorized_texts=vectorization_manager.vectorize_texts(tokenized_text,timestep_first=True)
    labels = dataset_manager.get_labels_in_batch('evaluation_set',step,batch_size)
    vectorized_labels=labels_manager.one_hot_vectorize(labels)
    
    _, accuracy_ = session.run(accuracy, feed_dict={tf_input: vectorized_texts, tf_output: vectorized_labels})

    results.append(accuracy_)

mean_= np.mean(results)
print('Accuracy: '+str(mean_), flush=True)

