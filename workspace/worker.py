from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import datetime
import json
import os

import numpy as np
import tensorflow as tf
from keras import backend
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export

import resnet as resnet

"""
Remember to set the TF_CONFIG envrionment variable.

For example:

export TF_CONFIG='{"cluster": {"worker": ["10.1.10.58:12345", "10.1.10.250:12345"]}, "task": {"index": 0, "type": "worker"}}'
"""

# communication_options = tf.distribute.experimental.CommunicationOptions(
#     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
# )
# strategy = tf.distribute.MultiWorkerMirroredStrategy(
#     communication_options=communication_options
# )
## - or -
strategy = tf.distribute.MultiWorkerMirroredStrategy()

NUM_GPUS = 2
BS_PER_GPU = 128
NUM_EPOCHS = 60

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


def normalize(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


def schedule(epoch):
    initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
    learning_rate = initial_learning_rate
    for mult, start_epoch in LR_SCHEDULE:
        if epoch >= start_epoch:
            learning_rate = initial_learning_rate * mult
        else:
            break
    tf.summary.scalar("learning rate", data=learning_rate, step=epoch)
    return learning_rate


def load_dataset():
    dirname = "cifar-10-batches-py"
    origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(
        dirname,
        origin=origin,
        untar=True,
        file_hash="6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
        cache_dir="/workspace",
    )

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)


(x, y), (x_test, y_test) = load_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_dataset = (
    train_dataset.map(augmentation)
    .map(normalize)
    .shuffle(NUM_TRAIN_SAMPLES)
    .batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
)
test_dataset = test_dataset.map(normalize).batch(
    BS_PER_GPU * NUM_GPUS, drop_remainder=True
)


input_shape = (HEIGHT, WIDTH, NUM_CHANNELS)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

APP_NAME = os.getenv("APP_NAME")
app_dir = os.path.join("/workspace/tensorboard", APP_NAME)
log_dir = os.path.join(
    app_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1
)

with strategy.scope():
    model = resnet.resnet56(img_input=img_input, classes=NUM_CLASSES)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[tensorboard_callback],
)
