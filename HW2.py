#
# COMPSCI X433.7 - Machine Learning With TensorFlow
# Homework Assignment #2, June, 10, 2018
# Hakan Egeli
#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# suppress the tensorflow cpu related warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# to get consistent results during development, fix the random seed
np.random.seed(3)

# input layer with with a place holder
with tf.name_scope("input"):
    a = tf.placeholder(tf.float32, shape=None, name="a")

# middle section
with tf.name_scope("middle_section"):
    b = tf.reduce_prod(a, name="b")
    c = tf.reduce_mean(a, name="c")
    d = tf.reduce_sum(a, name="d")
    e = tf.add(b, c, name="e")

# final node
with tf.name_scope("final_node"):
    f = tf.multiply(e, d, name="f")

# run the model
with tf.Session() as sess:
    # initialize variables
    tf.global_variables_initializer().run()

    # create an array of 100 normally distributed random numbers with Mean = 1 and Standard deviation = 2
    normal = np.random.normal(1.0, 2.0, 100)

    # run the final node and feed the 100 normally randomly distributed values as input to tensor a
    print(sess.run(f, feed_dict={ a: normal }))

    # write the graph
    sess.graph.as_graph_def()
    file_writer = tf.summary.FileWriter('./hw2_graph', sess.graph)

plt.hist(normal, bins=20)
plt.pause(1)
