# coding:utf-8

import os, sys
sys.path.append(os.pardir)
from misc.utils import leaky_relu

import tensorflow as tf
import sonnet as snt

class GeneratorDeconv(object):
    
    def __init__(self,
                 input_size = 100,
                 image_size = 64):
        self.input_size = input_size
        self.image_size = image_size
        self.name = 'generator'

    def __call__(self, inputs):
        with tf.variable_scope(self.name) as vs:
            x = snt.Linear(output_size = 512*int(self.image_size/16)**2)(inputs)
            x = snt.BatchNorm()(x, is_training = True)
            x = tf.nn.relu(x)
            x = tf.reshape(x, [-1, int(self.image_size/16), int(self.image_size/16), 512])
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

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    
class Discriminator(object):
    
    def __init__(self):
        self.name = 'discriminator'

    def __call__(self, inputs, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            x = snt.Conv2D(output_channels = 64, kernel_shape = (4, 4),
                           stride = (2, 2), padding = 'SAME')(inputs)
            x = leaky_relu(0.2)(x)
            x = snt.Conv2D(output_channels = 128, kernel_shape = (4, 4),
                           stride = (2, 2), padding = 'SAME')(inputs)
            x = snt.BatchNorm()(x, is_training = True)
            x = leaky_relu(0.2)(x)
            x = snt.Conv2D(output_channels = 256, kernel_shape = (4, 4),
                           stride = (2, 2), padding = 'SAME')(inputs)
            x = snt.BatchNorm()(x, is_training = True)
            x = leaky_relu(0.2)(x)
            x = snt.Conv2D(output_channels = 512, kernel_shape = (4, 4),
                           stride = (2, 2), padding = 'SAME')(inputs)
            x = snt.BatchNorm()(x, is_training = True)
            x = leaky_relu(0.2)(x)
            x = snt.BatchFlatten()(x)
            outputs = snt.Linear(output_size = 1)(x)
            return outputs

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
