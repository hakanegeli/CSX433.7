import os
import numpy as np
import tensorflow as tf
import csv
import tarfile
from sklearn.model_selection import train_test_split
from pandas import read_csv


class ImageData:
    """
    This static class is used to load the Blood Cell images and to prepare the training and test datasets
    """

    def check_and_uncompress_images(image_path='data/images'):
        error = False
        if not os.path.exists(image_path):
            print("Error! Image data path '{}' does not exist".format(image_path))
            error = True

        folder_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        for folder in folder_names:
            if not os.path.exists(os.path.join(image_path, folder)):
                if not os.path.isfile(os.path.join(image_path, "{}.tar.gz".format(folder))):
                    error = True
                    print("Image folder for {} and a .tar.gz archive cannot be found!".format(folder))
                else:
                    print("Extracting images from the tar file for {}".format(folder))
                    tar = tarfile.open(os.path.join(image_path, "{}.tar.gz".format(folder)))
                    tar.extractall(path=image_path)
                    tar.close()
                    if not os.path.exists(os.path.join(image_path, folder)):
                        print("Error extracting images from the tar file for {}".format(folder))
                        error = True
            else:
                print("Skipping extraction for {}...".format(folder))

        if error:
            exit(0)

    def image_names_and_labels_from_csv(path, filename):
        df = read_csv(os.path.join(path, filename), index_col=False, header=0)
        return df['image_name'].values, df['label'].values

    def image_names_and_labels(image_path='data/images', verbose=True):
        full_image_path = list()
        image_label = list()

        num_samples = 0
        for cell in os.listdir(image_path):
            if os.path.isdir(os.path.join(image_path, cell)):
                num_cells = len(os.listdir(os.path.join(image_path, cell)))
                num_samples += num_cells
                if verbose:
                    print('Cell: {:15s}  num samples: {:d}'.format(cell, num_cells))

                img_names = [a for a in os.listdir(os.path.join(image_path, cell)) if a.endswith('.jpeg')]
                img_paths = [os.path.join(image_path, cell, img_name) for img_name in img_names]
                full_image_path += img_paths
                if cell == "EOSINOPHIL":
                    image_label += list(np.zeros(len(img_paths)))
                elif cell == "LYMPHOCYTE":
                    image_label += list(np.ones(len(img_paths)))
                elif cell == "MONOCYTE":
                    image_label += list(np.ones(len(img_paths)) * 2)
                elif cell == "NEUTROPHIL":
                    image_label += list(np.ones(len(img_paths)) * 3)

        if verbose:
            print('Total samples from images folder: {:d}\n'.format(num_samples))

        image_label = np.int32(image_label)
        return full_image_path, image_label

    def train_dataset_input_fn(labels, images, batch_size, num_epochs, prefetch_to_device=True):
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

        # next 4 lines are slow for the first batch and fast for the rest of the batches
        # dataset = dataset.map(map_image)
        # dataset = dataset.shuffle(buffer_size=12500)
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.repeat(num_epochs)

        # following 4 lines are faster for the fisrt batch but slower for the following batches
        # dataset = dataset.shuffle(buffer_size=12500)
        # dataset = dataset.repeat(num_epochs)
        # dataset = dataset.map(map_image)
        # dataset = dataset.batch(batch_size)

        # # according to Tensorflow data pipeline folks, this is the best way to set the iterator!
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(12500, num_epochs))
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_image, batch_size))
        if prefetch_to_device:
            dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

        iterator = dataset.make_initializable_iterator()

        return iterator

    def test_dataset_input_fn(labels, images, batch_size, prefetch_to_device=True):
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
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_image, batch_size))
        if prefetch_to_device:
            dataset = dataset.apply(tf.contrib.data.prefetch_to_device("/gpu:0"))

        iterator = dataset.make_initializable_iterator()

        return iterator

    def stratify(x, y, test_size=0.1, shuffle=True):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y, shuffle=shuffle)
        return X_train, X_test, y_train, y_test

    def save_as_csv(x, y, headerrow, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), "w+") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(headerrow)
            writer.writerows(zip(x, y))


if __name__ == "__main__":
    x, y = ImageData.image_names_and_labels()
    print("x len: {}, y len: {}".format(len(x), len(y)))
