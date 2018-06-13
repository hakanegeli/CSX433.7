import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import ceil
from sklearn.model_selection import train_test_split

# suppress the tensorflow cpu related warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_epochs = 2
batch_size = 128

def image_names_and_labels():

    image_path = 'data/dataset2-master/images'

    x = list()
    y = list()

    print('Training samples:')
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
        else:
            y += list(np.ones(len(img_paths)) * 3)

    print('Total training samples: {:d}\n'.format(num_samples))

    print('Test samples:')
    test_dir = os.path.join(image_path, "TEST")
    num_samples = 0
    for cell in os.listdir(test_dir):
        num_cells = len(os.listdir(os.path.join(test_dir, cell)))
        num_samples += num_cells
        print('Cell: {:15s}  num samples: {:d}'.format(cell, num_cells))
    print('Total test samples: {:d}'.format(num_samples))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, shuffle=True)

    return X_train, X_test, y_train, y_test


def dataset_input_fn(labels, images):
    # Reads an image from a file,
    # decodes it into a dense tensor,
    # and resizes it to a fixed shape.
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [80, 60])
        return image_resized, label


    # A vector of filenames.
    images = tf.constant(images)

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(_parse_function)

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_initializable_iterator()

    return iterator


def main(argv):
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        X_train, X_test, y_train, y_test = image_names_and_labels()
        print(len(X_train), len(X_test), len(y_train), len(y_test))

        number_of_batches = int(ceil(len(y_train)/batch_size))
        print("{} batches in each epoch".format(number_of_batches))

        # images = tf.placeholder(tf.string, shape=[None])
        # labels = tf.placeholder(tf.float32, shape=[None])

        labels = y_train
        images = X_train

        iterator = dataset_input_fn(labels, images)
        next_batch = iterator.get_next()

        sess.run(iterator.initializer)

        iter = 1
        ep = 1
        while True:
            try:
                features, labels = sess.run(next_batch)
                print("iter = {}".format(iter))
                print("label = %s" % len(labels))
                print("features {}".format(features.shape))

                if (iter % number_of_batches == 0):
                    print("Epoc {} completed".format(ep))
                    ep += 1

                iter += 1
            except tf.errors.OutOfRangeError:
                break

        print("out of the while loop")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
