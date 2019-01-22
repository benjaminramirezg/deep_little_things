import numpy as np
import tensorflow as tf
from PrepareText import DatasetManager
from PrepareText import TokenizationManager
from PrepareText import VectorizationManager
from PrepareText import LabelsManager

# 1 - Usar como activaction function de la última capa sigmoid en vez de softmax (para que las probabilidades sean independientes entre cada clase)

# 2 - Usar como función de coste 'binary crossentropy', para penalizar independientemente en cada clase

# 3 - Convertir a tensor de unos y ceros con esto:

# NN HYPERPARAMETERS

num_epochs = 5000
batch_size=20
input_size=300
output_size=2
hidden_states_sizes=[128,128,64]
hidden_state_size=64
learning_rate=0.001
epsilon=0.1
threshold=0.5

# GRAPH DEFINITION

tf.reset_default_graph()

tf_input=tf.placeholder(shape=[None,None,input_size], dtype=tf.float32, name='tf_input')
tf_output=tf.placeholder(shape=[None,output_size], dtype=tf.float32, name='tf_output')

rnn_cells_fw=[tf.contrib.rnn.LSTMCell(num_units=n) for n in hidden_states_sizes]
rnn_cells_bw=[tf.contrib.rnn.LSTMCell(num_units=n) for n in hidden_states_sizes]
##rnn_cells=tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=n) for n in hidden_states_sizes])

outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(rnn_cells_fw, rnn_cells_bw, tf_input, dtype=tf.float32, time_major=True)
##outputs, state = tf.nn.dynamic_rnn(rnn_cells, tf_input, dtype=tf.float32, time_major=True)
output=tf.layers.dense(outputs[-1,:,:], output_size)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_output, logits=output)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sigmoid=tf.nn.sigmoid(output)
op_predictions = tf.where(tf.greater(sigmoid, threshold), tf.ones_like(sigmoid), tf.zeros_like(sigmoid), name='op_predictions')
accuracy = tf.metrics.accuracy(tf_output, predictions=op_predictions)

# CREATING SAVER

saver = tf.train.Saver()

# STARTING SESSION

session=tf.InteractiveSession()

# RUNNING SESSION

session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

# CREATING CLASSES

dataset_manager=DatasetManager(file_name='../data/many2multilabel.csv',file_format='csv', text_position=2, labels_first_position=0, labels_last_position=1, evaluation_set_percentage = 20)
tokenization_manager=TokenizationManager(tokenizer='gensim')
vectorization_manager=VectorizationManager(vector_size=input_size,embeddings_file='../embeddings/wiki.en.bin', binary_file=True)
labels_manager=LabelsManager(["turn_on_light","set_alarm"])

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
        vectorized_labels=labels
        
        _, loss_ = session.run([train_op,loss], {tf_input: vectorized_texts, tf_output: vectorized_labels})

        if step % 10 == 0:
            print(str(step) + ' batches (Loss: '+ str(loss_) + ')', flush=True)

# SAVING MODEL

saver.save(session, './model-many2multilabel')

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
    vectorized_labels=labels
    
    _, accuracy_ = session.run(accuracy, feed_dict={tf_input: vectorized_texts, tf_output: vectorized_labels})

    results.append(accuracy_)

mean_= np.mean(results)
print('Accuracy: '+str(mean_), flush=True)
