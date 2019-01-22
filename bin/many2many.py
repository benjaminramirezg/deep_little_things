import numpy as np
import tensorflow as tf
from PrepareText import CoNLLManager
from PrepareText import TokenizationManager
from PrepareText import VectorizationManager
from PrepareText import LabelsManager

num_epochs = 1000
batch_size=10
input_size=300
output_size=5
hidden_states_sizes=[128,128,64]
hidden_state_size=64
learning_rate=0.001
epsilon=0.1
threshold=0.5

# GRAPH DEFINITION

tf.reset_default_graph()

tf_input=tf.placeholder(shape=[None,None,input_size], dtype=tf.float32, name='tf_input')
tf_output=tf.placeholder(shape=[None,None,output_size], dtype=tf.float32, name='tf_output')

rnn_cells=tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=n) for n in hidden_states_sizes])
outputs, state = tf.nn.dynamic_rnn(rnn_cells, tf_input, dtype=tf.float32, time_major=True)
output=tf.layers.dense(outputs, output_size)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_output, logits=output)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

op_predictions = tf.argmax(tf.nn.softmax(output), axis=2, name='op_predictions')
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_output, axis=2), predictions=op_predictions)

# CREATING SAVER

saver = tf.train.Saver()

# STARTING SESSION

session=tf.InteractiveSession()

# RUNNING SESSION

session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

# CREATING CLASSES

dataset_manager=CoNLLManager(file_name='../data/many2many.txt',sentence_break='end',evaluation_set_percentage = 20)
tokenization_manager=TokenizationManager(tokenizer='none')
vectorization_manager=VectorizationManager(vector_size=input_size,binary_file=True,embeddings_file='../embeddings/wiki.en.bin')
labels_manager=LabelsManager(["ACTION-B","ACTION-I","OBJECT-B","OBJECT-I","NONE"])

for epoch in range(num_epochs):
    print('', flush=True)
    print('=== ' + str(epoch) + ' epochs ===', flush=True)
    print('', flush=True)    
    for step in range(len(dataset_manager.training_set)//batch_size):

        raw_texts = dataset_manager.get_texts_in_batch('training_set',step,batch_size)
        tokenized_texts=tokenization_manager.tokenize_texts(raw_texts)
        vectorized_texts=vectorization_manager.vectorize_texts(tokenized_texts,timestep_first=True)
        labels = dataset_manager.get_labels_in_batch('training_set',step,batch_size)
        vectorized_labels=labels_manager.one_hot_vectorize_multiple_steps(labels,timestep_first=True)
        _, loss_ = session.run([train_op,loss], {tf_input: vectorized_texts, tf_output: vectorized_labels})

        if step % 10 == 0:
            print(str(step) + ' batches (Loss: '+ str(loss_) + ')', flush=True)

# SAVING MODEL

saver.save(session, './model-many2many')

# EVALUATION

print('', flush=True)
print('==============', flush=True)
print('= EVALUATION =', flush=True)
print('==============', flush=True)

results=[]

for step in range(len(dataset_manager.evaluation_set)//batch_size):

    raw_texts=dataset_manager.get_texts_in_batch('evaluation_set',step,batch_size)
    tokenized_text=tokenization_manager.tokenize_texts(raw_texts)
    vectorized_texts=vectorization_manager.vectorize_texts(tokenized_text,timestep_first=True)
    labels = dataset_manager.get_labels_in_batch('evaluation_set',step,batch_size)
    vectorized_labels=labels_manager.one_hot_vectorize_multiple_steps(labels,timestep_first=True)
    
    _, accuracy_ = session.run(accuracy, feed_dict={tf_input: vectorized_texts, tf_output: vectorized_labels})

    results.append(accuracy_)

mean_= np.mean(results)
print('Accuracy: '+str(mean_), flush=True)

