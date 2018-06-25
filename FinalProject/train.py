import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import time
from plot import plot_data
from etl import ImageData
from math import ceil

# suppress the tensorflow cpu related warning
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
COMPSCI X433.7 - Machine Learning With TensorFlow
Final Project, June, 26, 2018
Hakan Egeli
 
 
This module conatins the following functionality: 
* model definition for the Convolutional Neural Network,
* code to train and test the model
* code to save the model graph and various parameters to be viewed in Tensorboard
* console log of the training/test progress with a plot of the Model Loss over the iterations
"""

LOG_DIR = 'model'
num_epochs = 5
batch_size = 128


# define the model
def cnn_model_fn(features, is_training, reuse=tf.AUTO_REUSE):
    # Input Layer
    with tf.name_scope("input_layer"):
        input_layer = tf.convert_to_tensor(tf.reshape(features, [-1, 160, 120, 3]), name="input")

    # Convolutional Layer #1
    with tf.name_scope("conv_layer_1"):
        layer = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            reuse=reuse,
            name="conv1")
        tf.summary.histogram("conv", layer)
        layer = tf.layers.max_pooling2d(
            inputs=layer,
            pool_size=[2, 2],
            strides=2,
            name="pool1")
        tf.summary.histogram("pool", layer)
        layer = tf.layers.batch_normalization(layer, training=is_training, reuse=reuse, name="bm1")
        tf.summary.histogram("batch_norm", layer)
        layer = tf.layers.dropout(inputs=layer, rate=0.2, training=is_training, name="dropout1")
        tf.summary.histogram("dropout", layer)

    # Convolutional Layer #2
    with tf.name_scope("conv_layer_2"):
        layer = tf.layers.conv2d(
            inputs=layer,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            reuse=reuse,
            name="conv2")
        tf.summary.histogram("conv", layer)
        layer = tf.layers.max_pooling2d(
            inputs=layer,
            pool_size=[2, 2],
            strides=2,
            name="pool2")
        tf.summary.histogram("pool", layer)
        layer = tf.layers.batch_normalization(layer, training=is_training, reuse=reuse, name="bn2")
        tf.summary.histogram("batch_norm", layer)
        layer = tf.layers.dropout(inputs=layer, rate=0.2, training=is_training, name="dropout2")
        tf.summary.histogram("dropout", layer)

    # Convolutional Layer #3
    with tf.name_scope("conv_layer_3"):
        layer = tf.layers.conv2d(
            inputs=layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            reuse=reuse,
            name="conv3")
        tf.summary.histogram("conv", layer)
        layer = tf.layers.max_pooling2d(
            inputs=layer,
            pool_size=[2, 2],
            strides=2,
            name="pool3")
        tf.summary.histogram("pool", layer)
        layer = tf.layers.batch_normalization(layer, training=is_training, reuse=reuse, name="bn3")
        tf.summary.histogram("batch_norm", layer)
        layer = tf.layers.dropout(inputs=layer, rate=0.2, training=is_training, name="dropout3")
        tf.summary.histogram("dropout", layer)

    # Convolutional Layer #4
    with tf.name_scope("conv_layer_4"):
        layer = tf.layers.conv2d(
            inputs=layer,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            reuse=reuse,
            name="conv4")
        tf.summary.histogram("conv", layer)
        layer = tf.layers.max_pooling2d(
            inputs=layer,
            pool_size=[2, 2],
            strides=2,
            name="pool4")
        tf.summary.histogram("pool", layer)
        layer = tf.layers.batch_normalization(layer, training=is_training, reuse=reuse, name="bn4")
        tf.summary.histogram("batch_norm", layer)
        layer = tf.layers.dropout(inputs=layer, rate=0.2, training=is_training, name="dropout4")
        tf.summary.histogram("dropout", layer)

    # Fully Connected Layer #1
    with tf.name_scope("fully_connected_layer_1"):
        layer = tf.reshape(layer, [-1, 10 * 7 * 64])
        layer = tf.layers.dense(
            inputs=layer,
            units=128,
            activation=tf.nn.relu,
            reuse=reuse,
            name="dense1")
        tf.summary.histogram("fc", layer)
        layer = tf.layers.batch_normalization(layer, training=is_training, reuse=reuse, name="bn5")
        tf.summary.histogram("batch_norm", layer)
        layer = tf.layers.dropout(inputs=layer, rate=0.2, training=is_training, name="dropout5")
        tf.summary.histogram("dropout", layer)

    # Logits Layer
    with tf.name_scope("logits_layer"):
        logits = tf.layers.dense(inputs=layer, units=4, name="logits")
        tf.summary.histogram("logits", logits)

    return logits


# loss function for the model
def loss_fn(labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


# we will use accuracy to evaluate the performance of the training
def accuracy_fn(predictions):
    return tf.reduce_mean(tf.cast(predictions, tf.float32))


# main entry to the code
def main(argv):
    tf.reset_default_graph()

    g = tf.Graph()
    with g.as_default():
        # check if the image folders exist. If the image folders do not exist this call will unpack the .tar.gz files
        ImageData.check_and_uncompress_images()

        # read all the images from the image folder, ang get the image names and labels
        image_names, image_labels = ImageData.image_names_and_labels()

        # images that are read from the folders need to be stratified.
        # shuffle and split the dataset into 90%-10% train/test.
        # stratify will make sure that the classes are distributed equally among these two sets
        X_train, X_test, y_train, y_test = ImageData.stratify(image_names, image_labels, test_size=0.1, shuffle=True)

        train_count = len(y_train)
        test_count = len(y_test)
        print("Train Image Count = {}\nTest Image Count = {}".format(train_count, test_count))

        # saving the reference to the test images for later inference!
        ImageData.save_as_csv(X_test, y_test, ['image_name', 'label'], os.path.join('data', 'test_data.csv'))

        number_of_batches = int(ceil(len(y_train) / batch_size))
        print("{} batches in each epoch\n".format(number_of_batches))

        # training data pipeline and the iterator
        train_images = X_train
        train_labels = tf.cast(y_train, tf.int32)

        train_iterator = ImageData.train_dataset_input_fn(train_labels, train_images, batch_size, num_epochs)
        train_next_batch = train_iterator.get_next()

        # test data pipeline and the iterator
        test_images = X_test
        test_labels = tf.cast(y_test, tf.int32)

        test_iterator = ImageData.test_dataset_input_fn(test_labels, test_images, batch_size)
        test_next_batch = test_iterator.get_next()

        # embedding data pipeline and the iterator
        # we have selected 1/4th of the Test Data, 311 rows, due to the memory limitations of our GPU
        embedding_iterator = ImageData.test_dataset_input_fn(test_labels, test_images, 311)
        embedding_next_batch = embedding_iterator.get_next()

        # placeholders
        features = tf.placeholder(tf.float32, shape=(None, 160, 120, 3))
        labels = tf.placeholder(tf.int32, shape=(None,))

        # model
        logits = cnn_model_fn(features, is_training=True)

        # embedding
        # we will use the last layer for embeddings (for Projector visualization in tensorboard)
        embedding = tf.Variable(np.zeros([311, logits.shape[1]]), dtype=tf.float32, name='test_embedding')
        assignment = embedding.assign(logits)

        metadata = 'metadata.tsv'
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name
        embedding_config.metadata_path = metadata

        # loss
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                loss = loss_fn(labels, logits)
            tf.summary.scalar('loss', loss)
            tf.summary.histogram("loss", loss)

        # optimizer
        with tf.name_scope('optimizer'):
            lr = 0.1
            step_rate = 1000
            decay = 0.95

            global_step = tf.Variable(0, trainable=False)
            tf.assign(global_step, global_step + 1)

            learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01)

            train_op = optimizer.minimize(
                loss=loss,
                global_step=global_step)

        # predictions
        with tf.name_scope('predictions'):
            predictions = tf.cast(tf.argmax(input=logits, axis=1), tf.int32)

        # model performance metric
        with tf.name_scope('performance'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(predictions, labels)
            with tf.name_scope('accuracy'):
                accuracy = accuracy_fn(correct_prediction)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram("accuracy", accuracy)

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()

        start_time = time.time()
        with tf.Session(graph=g) as sess:

            sess.graph.as_graph_def()

            # we will log training values separately from the test values
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(test_writer, config)

            # merge all sumaries so that they can be written to the model file
            merged_summaries = tf.summary.merge_all()

            # initializations
            sess.run(init)
            sess.run(train_iterator.initializer)
            sess.run(test_iterator.initializer)
            sess.run(embedding_iterator.initializer)

            iter = 0
            epoch = 1
            l = list()
            # iterate until the batch iterator finishes all the batches and epochs
            while True:
                try:
                    # get the next batch of training data...
                    train_next_features, train_next_labels = sess.run(train_next_batch)
                    feed_dict_train = {features: train_next_features,
                                       labels: train_next_labels}

                    # ...and train the model
                    with tf.name_scope('training'):
                        loss_train, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)
                        l.append(loss_train)

                    # at every 20 iterations, calculate the accuracy and log the summaries for both tran and test
                    if iter % 20 == 0:
                        with tf.name_scope('evaluation'):
                            with tf.name_scope('train'):
                                train_batch_predictions, summaries, accuracy_train = sess.run(
                                    [predictions, merged_summaries, accuracy],
                                    feed_dict=feed_dict_train)

                                train_writer.add_summary(summaries, global_step=iter)

                            test_next_features, test_next_labels = sess.run(test_next_batch)
                            feed_dict_test = {features: test_next_features,
                                              labels: test_next_labels}
                            with tf.name_scope('test'):
                                # get the next test batch which the model hasn't seen!
                                test_batch_predictions, summaries, accuracy_test = sess.run(
                                    [predictions, merged_summaries, accuracy],
                                    feed_dict=feed_dict_test)

                                test_writer.add_summary(summaries, global_step=iter)

                                elapsed_time = time.time() - start_time
                                start_time = time.time()

                                print(
                                    "iter={:4d}, TRAIN loss={:5.2f}, acc={:5.2f}, TEST acc={:5.2f}, time={:5.2f}".format(
                                        iter,
                                        loss_train,
                                        accuracy_train,
                                        accuracy_test,
                                        elapsed_time))

                        save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), iter)

                    if iter > 0 and iter % number_of_batches == 0:
                        print("Epoch {} completed".format(epoch))
                        epoch += 1

                    iter += 1
                except tf.errors.OutOfRangeError:
                    # end of all epochs
                    print("Last Epoch {} completed".format(epoch))

                    # calculate the final test accuracy and model summaries and save it
                    test_next_features, test_next_labels = sess.run(test_next_batch)
                    feed_dict_test = {features: test_next_features,
                                      labels: test_next_labels}

                    test_batch_predictions, summaries, accuracy_test = sess.run(
                        [predictions, merged_summaries, accuracy],
                        feed_dict=feed_dict_test)

                    test_writer.add_summary(summaries, global_step=iter)

                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    print(
                        "iter={:4d}, FINAL TEST acc={:5.2f}, time={:5.2f}".format(
                            iter,
                            accuracy_test,
                            elapsed_time))

                    # calculate the embedding for Projector visualization in Tensorboard
                    print("Processing Embeddings")
                    embed_next_features, embed_next_labels = sess.run(embedding_next_batch)
                    feed_dict_embed = {features: embed_next_features,
                                       labels: embed_next_labels}

                    # and write the labels that match the input data to the metadata file
                    with open(os.path.join(LOG_DIR, 'test', metadata), 'w') as metadata_file:
                        for row in embed_next_labels:
                            metadata_file.write('%d\n' % row)

                    sess.run(assignment, feed_dict=feed_dict_embed)

                    elapsed_time = time.time() - start_time

                    print("Processing time={:5.2f}".format(elapsed_time))

                    # save the final checkpoint!
                    save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'), iter)
                    break

            print("done!")

            # plot the accumulated loss values over the course of iterations
            plot_data(np.arange(0, len(l), 1, dtype=np.int), l, num_epochs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
