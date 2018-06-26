import tensorflow as tf
import time
from plot import plot_confusion_matrix
from etl import ImageData
from sklearn.metrics import confusion_matrix, classification_report


# suppress the tensorflow cpu related warning
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
COMPSCI X433.7 - Machine Learning With TensorFlow
Final Project, June, 26, 2018
Hakan Egeli


This module conatins the following functionality: 
* model definition for the Convolutional Neural Network,
* code to restore the model from the checkpoint file
* code to evaluate the model
* console log of the test accuracy with a plot of the Model Confusion Matrix
"""

LOG_DIR = 'model'
num_epochs = 3
batch_size = 1244 # we are going to use all the images in the test set during evaluation


# define the model
def cnn_model_fn(features, is_training, reuse=tf.AUTO_REUSE):
    with tf.device('/cpu:0'):
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
        # we are going to use the image names that we saved when we stratified the data and created our test set
        # these images were not used to train the model!
        X_test, y_test = ImageData.image_names_and_labels_from_csv(path=LOG_DIR, filename='test_data.csv')

        test_count = len(y_test)
        print("Test Image Count = {}".format(test_count))

        # test data pipeline and the iterator
        test_images = X_test
        test_labels = tf.cast(y_test, tf.int32)

        test_iterator = ImageData.test_dataset_input_fn(test_labels, test_images, batch_size, prefetch_to_device=False)
        test_next_batch = test_iterator.get_next()

        # placeholders
        features = tf.placeholder(tf.float32, shape=(None, 160, 120, 3))
        labels = tf.placeholder(tf.int32, shape=(None,))

        # model
        logits = cnn_model_fn(features, is_training=True)

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

            learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay, staircase=True)

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

            merged_summaries = tf.summary.merge_all()

            # initializations
            sess.run(init)
            sess.run(test_iterator.initializer)

            # restore the session from the last saved checkpoint file
            iter = 0

            ckpt = tf.train.get_checkpoint_state(os.path.join('.', LOG_DIR))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                iter = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
                print("Session restored at iter={}".format(iter))

            # fetch a batch from the test data pipeline, obtain the predictions and evaluate the accuracy
            # this batch contains all the test images (see batch_size initialization at the top of the code)
            test_next_features, test_next_labels = sess.run(test_next_batch)
            feed_dict_eval = {features: test_next_features,
                              labels: test_next_labels}

            y_predictions, accuracy_test = sess.run(
                [predictions, accuracy],
                feed_dict=feed_dict_eval)

            elapsed_time = time.time() - start_time
            start_time = time.time()

            print(
                "iter={:4d}, FINAL TEST acc={:5.2f}, time={:5.2f}".format(
                    iter,
                    accuracy_test,
                    elapsed_time))

            # Accuracy is not enough to evaluate the model!
            # We will examine the Confusion Matrix to see how our model did predicting different classes
            # and calculate the following:
            #
            # Precision (the ability of the classifier not to label as positive a sample that is negative),
            # Recall (the ability of the classifier to find all the positive samples),
            # and F1-score values (weighted harmonic mean of the precision and recall)
            #
            cm = confusion_matrix(y_test, y_predictions)

            class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
            print(classification_report(y_test, y_predictions, target_names=class_names))
            plot_confusion_matrix(cm, classes=class_names)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
