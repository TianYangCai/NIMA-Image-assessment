import numpy as np
import os
import glob

import tensorflow as tf

base_images_path = 'Z:\\ava dataset\\AVA_dataset\\images\\images\\'
ava_dataset_path = 'Z:\\ava dataset\\AVA_dataset\\AVA.txt'

IMAGE_SIZE = 224

files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

train_image_paths = []
train_scores = []

print("Loading training set and val set")
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])

        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = base_images_path + str(id) + '.jpg'
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)

        count = 255000 // 20
        if i % count == 0 and i != 0:
            print('Loaded %d percent of the dataset' % (i / 255000. * 100))

train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')

val_image_paths = train_image_paths[-5000:]
val_scores = train_scores[-5000:]
train_image_paths = train_image_paths[:-5000]
train_scores = train_scores[:-5000]

print('Train set size : ', train_image_paths.shape, train_scores.shape)
print('Val set size : ', val_image_paths.shape, val_scores.shape)
print('Train and validation datasets ready !')

def parse_data(filename, scores):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (256, 256))
    image = tf.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def parse_data_without_augmentation(filename, scores):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def train_generator(batchsize, shuffle=True):
    with tf.Session() as sess:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)

def val_generator(batchsize):
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_scores))
        val_dataset = val_dataset.map(parse_data_without_augmentation)

        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()

        val_batch = val_iterator.get_next()

        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)

def features_generator(record_path, faeture_size, batchsize, shuffle=True):
    with tf.Session() as sess:

        def parse_single_record(serialized_example):
            example = tf.parse_single_example(
                serialized_example,
                features={
                    'features': tf.FixedLenFeature([faeture_size], tf.float32),
                    'scores': tf.FixedLenFeature([10], tf.float32),
                })

            features = example['features']
            scores = example['scores']
            return features, scores

        train_dataset = tf.data.TFRecordDataset([record_path])
        train_dataset = train_dataset.map(parse_single_record, num_parallel_calls=4)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=5)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)