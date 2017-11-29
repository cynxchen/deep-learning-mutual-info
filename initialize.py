import numpy as np
import scipy.io as sio
import os
import sys
import tensorflow as tf
import random
from calc_info import get_information, extract_array
import matplotlib.pyplot as plt

######################
# -- PREPARE DATA -- #
######################

# load data from file
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

# Split data in test and train
test = random.sample(xrange(4096), 1000)
train = ([i for i in xrange(4096) if i not in test])

train_data = data[train]
train_labels = labels[train]
test_data = data[test]
test_labels = labels[test]

# Check if the labels are proportional in the test and train groups
(train_labels[:,1] == 1).sum()
(train_labels[:,0] == 1).sum()
(test_labels[:,1] == 1).sum()
(test_labels[:,0] == 1).sum()


#############################################
# -- BUILD AND TRAIN DEEP NEURAL NETWORK -- #
#############################################
# based on tutorial code https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/?completed=/tensorflow-deep-neural-network-machine-learning-tutorial/

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
batch_points = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]


# build neural network
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
    output = tf.nn.softmax(output)
    layers = [l1, l2, l3, l4, l5, l6, l7, output]
    return layers


# get output values from each layer
def extract_activity(sess, layers):
    """Get the activation values of the layers for the input"""
    w_temp = []
    i=0
    for i in range(0, len(batch_points) - 1):
        batch_x = np.array(data[batch_points[i]:batch_points[i+1]])
        batch_y = np.array(labels[batch_points[i]:batch_points[i+1]])
        w_temp_local = sess.run([layers], feed_dict={x: batch_x, y: batch_y})
        # print(w_temp_local)
        for s in range(len(w_temp_local[0])):
            if i == 0:
                w_temp.append(w_temp_local[0][s])
            else:
                w_temp[s] = np.concatenate((w_temp[s], w_temp_local[0][s]), axis=0)
    return w_temp


# train the neural network given a number of epochs
def train_neural_network(x, num_epochs):
    layers = neural_network_model(x)
    prediction = layers[-1]
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(prediction,1e-50,1.0)), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0004).minimize(cross_entropy)
    hm_epochs = num_epochs
    ws, estimted_label, gradients, infomration, models, weights = [[None] * num_epochs for _ in range(6)]
    tf.set_random_seed(1234)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            ws[epoch] = extract_activity(sess, layers)
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
            # print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy_test:',accuracy.eval({x:test_data, y:test_labels}),
            'Accuracy_train:',accuracy.eval({x:train_data, y:train_labels}))
    return ws


# plot information Plane
def plot_info_plane(i, I_XT_array, I_TY_array):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    plt.scatter(I_XT_array, I_TY_array, color=color)
    plt.title("Information Plane after " + str(i) + " Epochs")
    plt.ylim([0,1])
    plt.xlim([0,12])
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    plt.savefig("plots/final/snapshot"+str(i)+".png")
    plt.show()


#############################################
# -- RUN NEURAL NETWORK AND CREATE PLOTS -- #
#############################################

epochs_list = [2, 100, 250, 500, 1000, 5000]
for i in epochs_list:
    print("CURRENTLY ON EPOCH NUM", i)
    I_XT_array = np.array([])
    I_TY_array = np.array([])
    for j in range(5):
        print("repeat", j)
        network = train_neural_network(x, i)
        network_info = get_information(network, data, labels, i)
        network_info_squeezed = np.squeeze(np.array(network_info))
        I_XT = np.array(extract_array(network_info_squeezed, 'local_IXT'))
        I_XT_array = np.append(I_XT_array, I_XT)
        I_TY = np.array(extract_array(network_info_squeezed, 'local_ITY'))
        I_TY_array = np.append(I_TY_array, I_TY)
    plot_info_plane(i, I_XT_array, I_TY_array)
