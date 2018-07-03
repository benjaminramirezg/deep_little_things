import numpy as np
import tensorflow as tf
from PrepareText import DatasetManager
from PrepareText import TokenizationManager
from PrepareText import VectorizationManager

# NN HYPERPARAMETERS

num_epochs = 50
batch_size=100
input_size=300
output_size=6
hidden_state_size=64
learning_rate=0.01
threshold=0.5

############
### MAIN ###
############

# GRAPH DEFINITION

tf.reset_default_graph()

tf_input=tf.placeholder(shape=[None,None,input_size], dtype=tf.float32, name='tf_input')
tf_output=tf.placeholder(shape=[None,output_size], dtype=tf.float32, name='tf_output')

rnn_cell=tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(rnn_cell, tf_input, initial_state=None, dtype=tf.float32, time_major=True)
output=tf.layers.dense(outputs[-1,:,:], output_size)

loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf_output, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

op_predictions = tf.to_int32(tf.sigmoid(output), name='op_predictions')
accuracy = tf.metrics.accuracy(labels=tf_output, predictions=op_predictions)
precision = tf.metrics.precision(labels=tf_output, predictions=op_predictions)
recall = tf.metrics.recall(labels=tf_output, predictions=op_predictions)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# STARTING SESSION

session=tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=4)

# RUNNING SESSION

session.run(init_op)

# CREATING CLASSES


dataset_manager=DatasetManager('./train.csv',file_format='csv', text_position=1, labels_first_position=2)
dataset_manager.split_train_eval_dataset(evaluation_set_percentage = 20)
tokenization_manager=TokenizationManager(tokenizer='gensim')
vectorization_manager=VectorizationManager(vector_size=input_size)
#vectorization_manager.set_socket_as_embeddings_source(host='localhost', port=1234, buf=1000000)
vectorization_manager.set_file_as_embeddings_source('./embeddings.bin', binary_file=True)

# TRAINING

print('')
print('============')
print('= TRAINING =')
print('============')
print('')

for epoch in range(num_epochs):
    print('=== ' + str(epoch) + ' epochs ===')
    print('')    
    for step in range(dataset_manager.training_set.shape[0]//batch_size):

        raw_texts=dataset_manager.get_training_set_texts_in_batch(step,batch_size)
        tokenization=tokenization_manager.tokenize_texts(raw_texts)
        texts_in_batch=vectorization_manager.vectorize_texts(tokenization,timestep_first=True)
        labels_in_batch = dataset_manager.get_training_set_labels_in_batch(step,batch_size)

        _, loss_ = session.run([train_op,loss], {tf_input: texts_in_batch, tf_output: labels_in_batch})

        if step % 50 == 0:
            print(str(step) + ' batches (Loss: '+ str(loss_) + ')')
            saver.save(session, './model')

# EVALUATION

print('')
print('==============')
print('= EVALUATION =')
print('==============')
print('')

precisions, recalls = [], []
for step in range(dataset_manager.evaluation_set.shape[0]//batch_size):

    raw_texts=dataset_manager.get_evaluation_set_texts_in_batch(step,batch_size)
    tokenization=tokenization_manager.tokenize_texts(raw_texts)
    texts_in_eval=vectorization_manager.vectorize_texts(tokenization,timestep_first=True)
    labels_in_eval = dataset_manager.get_evaluation_set_labels_in_batch(step,batch_size)
    
    precision_, _ = session.run(precision, feed_dict={tf_input: texts_in_eval, tf_output: labels_in_eval})
    recall_, _ = session.run(recall, feed_dict={tf_input: texts_in_eval, tf_output: labels_in_eval})
    precisions.append(precision_)
    recalls.append(recall_)
    if step % 50 == 0:
        print(str(step) + ' batches (Precision: ' + str(precision_) + ' Recall: ' + str(recall_) + ')')

procesion_ = np.mean(precisions)
recall_ = np.mean(recalls)

print('')
print('Precision (mean) ' + str(precision_) + ' Recall (mean) ' + str(recall_))

#vectorization_manager.close_socket()
