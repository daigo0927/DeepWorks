# coding: utf-8

import tensorflow as tf
import tensorflow.contrib.layers as tcl

def bn_relu_conv(inputs, num_outputs, kernel_size, stride = (1, 1), padding = 'SAME'):
    x = tcl.batch_norm(inputs)
    x = tf.nn.relu(x)
    x = tcl.conv2d(x,
                   num_outputs = num_outputs,
                   kernel_size = kernel_size,
                   stride = stride,
                   padding = padding)
    return x
