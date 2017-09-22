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

# ConvTranspose (often called Deconv) ver.
def GeneratorDeconv(input_size = 100, image_size = 64): 

    w = int(image_size)

    inputs = Input(shape = (input_size, ))
    x = Dense(512*int(w/16)**2)(inputs) #shape(512*(w/16)**2,)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((int(w/16), int(w/16), 512))(x) # shape(w/16, w/16, 512)
    x = Conv2DTranspose(256, (4, 4), strides = (2, 2),
                        kernel_initializer = init,
                        padding = 'same')(x) # shape(w/8, w/8, 256)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (4, 4), strides = (2, 2),
                        kernel_initializer = init,
                        padding = 'same')(x) # shape(w/4, w/4, 128)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (4, 4), strides = (2, 2),
                        kernel_initializer = init,
                        padding = 'same')(x) # shape(w/2, w/2, 64)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(3, (4, 4), strides= (2, 2),
                        kernel_initializer = init,
                        padding = 'same')(x) # shape(w, w, 3)
    images = Activation('tanh')(x)

    model = Model(inputs = inputs, outputs = images)
    model.summary()
    return model


def Discriminator(image_size = 64, input_channel = 3):

    w = int(image_size)
    ch = int(input_channel)

    images = Input(shape = (w, w, input_channel))
    x = Conv2D(64, (4, 4), strides = (2, 2),
               kernel_initializer = init, padding = 'same')(images) # shape(w/2, w/2, 32)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides = (2, 2),
               kernel_initializer = init, padding = 'same')(x) # shape(w/4, w/4, 64)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides = (2, 2),
               kernel_initializer = init, padding = 'same')(x) # shape(w/8, w/8, 128)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (4, 4), strides = (2, 2),
               kernel_initializer = init, padding = 'same')(x) # shape(L/16, L/16, 256)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs = images, outputs = outputs)
    model.summary()
    return model
