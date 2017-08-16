# coding:utf-8

import os, sys
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, Add
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.initializers import RandomNormal

init = RandomNormal(mean = 0., stddev = 0.02)

def _bn_relu_conv(filters, kernel_size = (3, 3), strides = (1, 1)):
    def f(inputs):
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, strides = strides,
                   kernel_initializer = init, padding = 'same')(x)
        return x
    return f

def _shortcut(inputs, x):
    # shortcut path
    _, inputs_w, inputs_h, inputs_ch = K.int_shape(inputs)
    _, x_w, x_h, x_ch = K.int_shape(x)
    stride_w = int(round(inputs_w / x_w))
    stride_h = int(round(inputs_h / x_h))
    equal_ch = inputs_ch == x_ch
    

    if stride_w>1 or stride_h>1 or not equal_ch:
        shortcut = Conv2D(x_ch, (1, 1),
                          strides = (stride_w, stride_h),
                          kernel_initializer = init, padding = 'valid')(inputs)
    else:
        shortcut = inputs
        
    merged = Add()([shortcut, x])
    return merged

def plain_block(filters, subsample = (1, 1)):
    def f(inputs):
        # convolution path
        x = _bn_relu_conv(filters, strides = subsample)(inputs)
        x = _bn_relu_conv(filters)(x)
        
        return _shortcut(inputs, x) # merge
    return f

def bottleneck_block(filters, subsample = (1, 1)):
    def f(inputs):
        # convolution path
        x = _bn_relu_conv(filters, kernel_size = (1, 1),
                          strides = subsample)(inputs)
        x = _bn_relu_conv(filters, kernel_size = (3, 3))(x)
        x = _bn_relu_conv(filters*4, kernel_size = (1, 1))(x)

        return _shortcut(inputs, x) # merge
    return f

def _residual_block(block_fn, filters, repetitions, is_first_layer = False):
    def f(inputs):
        x = inputs
        for i in range(repetitions):
            subsample = (1, 1)
            if i == 0 and not is_first_layer:
                subsample = (2, 2)
            x = block_fn(filters, subsample)(x)
        return x
    return f

class ResNetBuilder:

    @staticmethod
    def build(input_shape, num_outputs,
              block_fn, repetitions):

        inputs = Input(shape = input_shape)
        conv1 = Conv2D(64, (7, 7), strides = (2, 2),
                       padding = 'same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2),
                            padding = 'same')(conv1)

        x = pool1
        filters = 64
        first_layer = True
        for i, r in enumerate(repetitions):
            x = _residual_block(block_fn, filters = filters,
                                repetitions = r, is_first_layer = first_layer)(x)
            filters *= 2
            if first_layer:
                first_layer = False

        # last activation <- unnecessary???
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        _, w, h, ch = K.int_shape(x)
        pool2 = AveragePooling2D(pool_size = (w, h), strides = (1, 1))(x)
        flat1 = Flatten()(pool2)
        outputs = Dense(num_outputs, kernel_initializer = init,
                        activation = 'softmax')(flat1)
        
        model = Model(inputs = inputs, outputs = outputs)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs,
                                   plain_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs,
                                   plain_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs,
                                   bottleneck_block, [3, 4, 6, 3])
    
    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs,
                                   bottleneck_block, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs,
                                   bottleneck_block, [3, 8, 36, 3])
        
                

            
            
