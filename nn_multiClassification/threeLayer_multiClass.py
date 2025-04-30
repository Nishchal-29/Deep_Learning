# to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings("ignore")

# importing required libraries
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time

# loading datasets
train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")

# loading training and test data
x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])
x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

# images_iter = iter(x_train)
# labels_iter = iter(y_train)
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     ax = plt.subplot(5, 5, i + 1)
#     plt.imshow(next(images_iter).numpy().astype("uint8"))
#     plt.title(next(labels_iter).numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()

# transform an image into a tensor of shape (64, 64, 3) and normalize its components

def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1,])
    return image

new_train = x_train.map(normalize)
new_test = x_test.map(normalize)


