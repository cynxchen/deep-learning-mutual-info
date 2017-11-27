import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
import os
import sys
import tensorflow as tf
import random


def load_data(name, random_labels=False):
	"""Load the data
	name - the name of the dataset
	random_labels - True if we want to return random labels to the dataset
	return object with data and labels"""
	C = type('type_C', (object,), {})
	data_sets = C()
	d = sio.loadmat(os.path.join(os.getcwd(), name + '.mat'))
	F = d['F']
	y = d['y']
	C = type('type_C', (object,), {})
	data_sets = C()
	data_sets.data = F
	data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
	return data_sets


dataset = load_data("var_u")
data = dataset.data
labels = dataset.labels

# EXPLORING

data.shape
data[0]
labels.shape
labels[0]

# train_data = data[:3000]
# train_labels = labels[:3000]
# test_data = data[3000:]
# test_labels = labels[3000:]

test = random.sample(xrange(4096), 1000)
train = ([i for i in xrange(4096) if i not in test])

train_data = data[train]
train_labels = labels[train]
test_data = data[test]
test_labels = labels[test]

(train_labels[:,1] == 1).sum()
(train_labels[:,0] == 1).sum()
(test_labels[:,1] == 1).sum()
(test_labels[:,0] == 1).sum()

#---------- tutorial code https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/?completed=/tensorflow-deep-neural-network-machine-learning-tutorial/

n_nodes_hl1 = 12
n_nodes_hl2 = 10
n_nodes_hl3 = 7
n_nodes_hl4 = 5
n_nodes_hl5 = 4
n_nodes_hl6 = 3
n_nodes_hl7 = 2
n_classes = 2
batch_size = 500
x = tf.placeholder(tf.float32, [None, 12])
y = tf.placeholder(tf.float32)

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal(
        [12, n_nodes_hl1], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl1)))),
                      'biases':tf.Variable(tf.constant(0.0, shape = [n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal(
        [n_nodes_hl1, n_nodes_hl2], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl2)))),
                      'biases':tf.Variable(tf.constant(0.0, shape = [n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal(
        [n_nodes_hl2, n_nodes_hl3], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl3)))),
                      'biases':tf.Variable(tf.constant(0.0, shape = [n_nodes_hl3]))}

    hidden_4_layer = {'weights':tf.Variable(tf.truncated_normal(
        [n_nodes_hl3, n_nodes_hl4], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl4)))),
                      'biases':tf.Variable(tf.constant(0.0, shape = [n_nodes_hl4]))}

    hidden_5_layer = {'weights':tf.Variable(tf.truncated_normal(
        [n_nodes_hl4, n_nodes_hl5], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl5)))),
                      'biases':tf.Variable(tf.constant(0.0, shape = [n_nodes_hl5]))}

    hidden_6_layer = {'weights':tf.Variable(tf.truncated_normal(
        [n_nodes_hl5, n_nodes_hl6], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl6)))),
                      'biases':tf.Variable(tf.constant(0.0, shape = [n_nodes_hl6]))}

    hidden_7_layer = {'weights':tf.Variable(tf.truncated_normal(
        [n_nodes_hl6, n_nodes_hl7], mean=0.0, stddev=1.0 / np.sqrt(float(n_nodes_hl7)))),
                      'biases':tf.Variable(tf.constant(0.0, shape = [n_nodes_hl7]))}

    output_layer = {'weights':tf.Variable(tf.truncated_normal(
        [n_nodes_hl7, n_classes], mean=0.0, stddev=1.0 / np.sqrt(float(n_classes)))),
                    'biases':tf.Variable(tf.constant(0.0, shape = [n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.tanh(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.tanh(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.tanh(l3)

    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.tanh(l4)

    l5 = tf.add(tf.matmul(l4,hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.tanh(l5)

    l6 = tf.add(tf.matmul(l5,hidden_6_layer['weights']), hidden_6_layer['biases'])
    l6 = tf.nn.tanh(l6)

    l7 = tf.add(tf.matmul(l6,hidden_7_layer['weights']), hidden_7_layer['biases'])
    l7 = tf.nn.tanh(l7)

    output = tf.matmul(l7,output_layer['weights']) + output_layer['biases']
    return tf.nn.softmax(output)

def train_neural_network(x, num_epochs):
    prediction = neural_network_model(x)
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y*tf.log(tf.clip_by_value(prediction,1e-50,1.0)), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0004).minimize(cross_entropy)
    hm_epochs = num_epochs
    tf.set_random_seed(1234)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_data):
                start = i
                end = i+batch_size
                batch_x = np.array(train_data[start:end])
                batch_y = np.array(train_labels[start:end])
                _, c= sess.run([optimizer, cross_entropy], feed_dict={x: batch_x,
                                                                        y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy_test:',accuracy.eval({x:test_data, y:test_labels}),
            'Accuracy_train:',accuracy.eval({x:train_data, y:train_labels}))

# np.random.seed(2)
train_neural_network(x, 10)
train_neural_network(x, 50)
train_neural_network(x, 100)
train_neural_network(x, 200)
train_neural_network(x, 500)
train_neural_network(x, 750)
train_neural_network(x, 1000)
train_neural_network(x, 2000)