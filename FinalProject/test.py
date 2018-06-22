import tensorflow as tf
import numpy as np
import time
# import matplotlib.pyplot as plt

from math import ceil
from sklearn.model_selection import train_test_split

# suppress the tensorflow cpu related warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_epochs = 5
batch_size = 128

def image_names_and_labels():

    image_path = 'data/dataset2-master/images'

    x = list()
    y = list()

    train_dir = os.path.join(image_path, "TRAIN")
    num_samples = 0
    for cell in os.listdir(train_dir):
        num_cells = len(os.listdir(os.path.join(train_dir, cell)))
        num_samples += num_cells
        print('Cell: {:15s}  num samples: {:d}'.format(cell, num_cells))

        img_names = [a for a in os.listdir(os.path.join(train_dir, cell)) if a.endswith('.jpeg')]
        img_paths = [os.path.join(train_dir, cell, img_name) for img_name in img_names]
        x += img_paths
        if cell == "EOSINOPHIL":
            y += list(np.zeros(len(img_paths)))
        elif cell == "LYMPHOCYTE":
            y += list(np.ones(len(img_paths)))
        elif cell == "MONOCYTE":
            y += list(np.ones(len(img_paths)) * 2)
        elif cell == "NEUTROPHIL":
            y += list(np.ones(len(img_paths)) * 3)

    print('Total samples from TRAINING folder: {:d}\n'.format(num_samples))

    test_dir = os.path.join(image_path, "TEST")
    num_samples = 0
    for cell in os.listdir(test_dir):
        num_cells = len(os.listdir(os.path.join(test_dir, cell)))
        num_samples += num_cells
        print('Cell: {:15s}  num samples: {:d}'.format(cell, num_cells))

        img_names = [a for a in os.listdir(os.path.join(test_dir, cell)) if a.endswith('.jpeg')]
        img_paths = [os.path.join(test_dir, cell, img_name) for img_name in img_names]
        x += img_paths
        if cell == "EOSINOPHIL":
            y += list(np.zeros(len(img_paths)))
        elif cell == "LYMPHOCYTE":
            y += list(np.ones(len(img_paths)))
        elif cell == "MONOCYTE":
            y += list(np.ones(len(img_paths)) * 2)
        elif cell == "NEUTROPHIL":
            y += list(np.ones(len(img_paths)) * 3)

    print('Total samples from TEST: {:d}'.format(num_samples))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, shuffle=True)

    return X_train, X_test, y_train, y_test


def train_dataset_input_fn(labels, images):
    # Reads an image from a file,
    # decodes it into a dense tensor,
    # and resizes it to a fixed shape.
    def map_image(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [160, 120])
        standard_image = tf.image.per_image_standardization(image_resized)
        return standard_image, label


    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # dataset = dataset.map(map_image)
    # dataset = dataset.shuffle(buffer_size=10000)
    # dataset = dataset.batch(batch_size)
    # dataset = dataset.repeat(num_epochs)

    # dataset = dataset.shuffle(buffer_size=10000)
    # dataset = dataset.repeat(num_epochs)
    # dataset = dataset.map(map_image)
    # dataset = dataset.batch(batch_size)

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, num_epochs))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_image, batch_size))

    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

    iterator = dataset.make_initializable_iterator()

    return iterator


def validate_dataset_input_fn(labels, images):
    # Reads an image from a file,
    # decodes it into a dense tensor,
    # and resizes it to a fixed shape.
    def map_image(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [160, 120])
        standard_image = tf.image.per_image_standardization(image_resized)
        return standard_image, label


    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.repeat()
    # dataset = dataset.map(map_image)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_image, batch_size))

    dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

    iterator = dataset.make_initializable_iterator()

    return iterator


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
            units=256,
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
        tf.summary.histogram("logits", layer)

    return logits


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(predictions == labels) /
      predictions.shape[0])


def main(argv):
    tf.reset_default_graph()

    X_train, X_test, y_train, y_test = image_names_and_labels()

    train_count = len(y_train)
    test_count = len(y_test)
    print("\nTrain Image Count = {}\nValidate Image Count = {}".format(train_count, test_count))

    number_of_batches = int(ceil(len(y_train)/batch_size))
    print("{} batches in each epoch".format(number_of_batches))

    # training data
    train_images = X_train
    train_labels = tf.cast(y_train, tf.int32)

    train_iterator = train_dataset_input_fn(train_labels, train_images)
    train_next_batch = train_iterator.get_next()

    # validation data
    validate_images = X_test
    validate_labels = tf.cast(y_test, tf.int32)

    validate_iterator = validate_dataset_input_fn(validate_labels, validate_images)
    validate_next_batch = validate_iterator.get_next()

    # placeholders
    features = tf.placeholder(tf.float32, shape=(None, 160, 120, 3))
    labels = tf.placeholder(tf.int32, shape=(None, ))

    # models
    logits = cnn_model_fn(features, is_training=True)

    # loss
    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('cross_entropy', loss)
    tf.summary.histogram("cross_entropy", loss)

    # optimizer
    lr = 0.1
    step_rate = 1000
    decay = 0.95

    global_step = tf.Variable(0, trainable=False)
    tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01)

    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step)

    # predictions
    predictions = tf.cast(tf.argmax(input=logits, axis=1), tf.int32)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(predictions, labels)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram("accuracy", accuracy)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    start_time = time.time()
    with tf.Session() as sess:

        sess.graph.as_graph_def()

        writer = tf.summary.FileWriter('./model', sess.graph)
        summaries = tf.summary.merge_all()

        # initializations
        sess.run(init)
        sess.run(train_iterator.initializer)
        sess.run(validate_iterator.initializer)

        # iterate until the batch iterator finishes all the batches and epochs
        iter = 0
        ep = 1
        while True:
            try:
                train_next_features, train_next_labels = sess.run(train_next_batch)
                feed_dict = {features: train_next_features,
                             labels: train_next_labels}

                if iter % 20 == 0:
                    train_batch_loss, train_batch_predictions, _, summ = sess.run([loss, predictions, train_op, summaries],
                                                                          feed_dict=feed_dict)

                    eval_next_features, eval_next_labels = sess.run(validate_next_batch)

                    feed_dict = {features: eval_next_features,
                                 labels: eval_next_labels}

                    eval_batch_predictions = sess.run(predictions, feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    print("iter={:4d}, batch_loss={:5.2f}, train_err_rate={:5.2f}, val_err_rate={:5.2f}, elapsed time={:5.2f}".format(iter,
                                                                            train_batch_loss,
                                                                            error_rate(train_batch_predictions,train_next_labels),
                                                                            error_rate(eval_batch_predictions,eval_next_labels),
                                                                            elapsed_time))

                    writer.add_summary(summ, global_step=iter)
                    save_path = saver.save(sess, "./model/model.ckpt")
                else:
                    sess.run(train_op, feed_dict=feed_dict)

                if iter > 0 and iter % number_of_batches == 0:
                    print("Epoch {} completed".format(ep))
                    ep += 1

                iter += 1
            except tf.errors.OutOfRangeError:
                # end of epochs
                writer.add_summary(summ, global_step=iter)
                save_path = saver.save(sess, "./model/model.ckpt")
                break

        print("done!")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
