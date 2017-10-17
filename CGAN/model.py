# coding:utf-8

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import h5py

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.initializers import RandomNormal

init = RandomNormal(mean = 0., stddev = 0.02)

import tensorflow as tf
import sonnet as snt

# ConvTranspose (often called Deconv) ver.
def GeneratorDeconv(input_size = 100, image_size = 64):

    def build(inputs):
        x = snt.Linear(output_size = 512*int(image_size/16)**2)(inputs)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = tf.reshape(x, [-1, int(image_size/16), int(image_size/16), 512])
        x = snt.Conv2DTranspose(output_channels = 256, kernel_shape = (4, 4),
                                stride = (2, 2), padding = 'SAME')(x)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = snt.Conv2DTranspose(output_channels = 128, kernel_shape = (4, 4),
                                stride = (2, 2), padding = 'SAME')(x)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = snt.Conv2DTranspose(output_channels = 64, kernel_shape = (4, 4),
                                stride = (2, 2), padding = 'SAME')(x)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = snt.Conv2DTranspose(output_channels = 3, kernel_shape = (4, 4),
                                stride = (2, 2), padding = 'SAME')(x)
        images = tf.nn.tanh(x)
        return images

    return snt.Module(build, name = 'generator')


def Discriminator():

    def build(inputs):
        x = snt.Conv2D(output_channels = 64, kernel_shape = (4, 4),
                       stride = (2, 2), padding = 'SAME')(inputs)
        x = tf.contrib.keras.layers.LeakyReLU(0.2)(x)
        x = snt.Conv2D(output_channels = 128, kernel_shape = (4, 4),
                       stride = (2, 2), padding = 'SAME')(inputs)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.contrib.keras.layers.LeakyReLU(0.2)(x)
        x = snt.Conv2D(output_channels = 256, kernel_shape = (4, 4),
                       stride = (2, 2), padding = 'SAME')(inputs)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.contrib.keras.layers.LeakyReLU(0.2)(x)
        x = snt.Conv2D(output_channels = 512, kernel_shape = (4, 4),
                       stride = (2, 2), padding = 'SAME')(inputs)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.contrib.keras.layers.LeakyReLU(0.2)(x)
        x = snt.BatchFlatten()(x)
        outputs = snt.Linear(output_size = 1)(x)
        return outputs

    return snt.Module(build, name = 'discriminator')

