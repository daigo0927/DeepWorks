# coding:utf-8

import numpy as np
import tensorflow as tf


import argparse

from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.losses import categorical_crossentropy
from keras.initializers import RandomNormal

init = RandomNormal(mean = 0., stddev = 0.02)

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

def load_cifar10():
    print('load cifar10 data ...')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    return (x_train, y_train), (x_test, y_test)

def build_simpleCNN(input_shape = (32, 32, 3), num_output = 10):

    h, w, nch = input_shape
    assert h == w, 'expect input shape (h, w, nch), h == w'

    images = Input(shape = (h, h, nch))
    x = Conv2D(64, (4, 4), strides = (1, 1),
               kernel_initializer = init, padding = 'same')(images)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Conv2D(128, (4, 4), strides = (1, 1),
               kernel_initializer = init, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(num_output, kernel_initializer = init,
                    activation = 'softmax')(x)

    model = Model(inputs = images, outputs = outputs)
    return model

class TrainCNN(object):

    def __init__(self):

        self.sess = tf.Session()
        K.set_session(self.sess)
        
        self._build_graph()
        self._load_data()
    
    def _build_graph(self):
        
        self.images = tf.placeholder(tf.float32, (None, 32, 32, 3), name = 'images')
        self.labels = tf.placeholder(tf.float32, (None, 10), name = 'labels')
        self.model = build_simpleCNN()
        
        self.preds = self.model(self.images)
        self.loss = tf.reduce_mean(categorical_crossentropy(self.labels, self.preds))
        self.accuracy = tf.reduce_mean(tf.reduce_sum(self.labels*self.preds, axis = 1))

        self.opt = tf.train.AdamOptimizer()\
                           .minimize(self.loss, var_list = self.model.trainable_weights)

        self.sess.run(tf.global_variables_initializer())

    def _load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10()

    def train(self, batch_size = 128, num_epochs = 20):
        data_size = self.x_train.shape[0]
        num_batches = int(data_size/batch_size)

        for e in np.arange(num_epochs):
            # shuffle training data index
            permute_idx = np.random.permutation(np.arange(data_size))
        
            for b in np.arange(num_batches):

                # get data batch
                x_batch = self.x_train[permute_idx[b*batch_size:(b+1)*batch_size]]
                y_batch = self.y_train[permute_idx[b*batch_size:(b+1)*batch_size]]

                self.sess.run(self.opt,
                              feed_dict = {self.images:x_batch, self.labels:y_batch,
                                                     K.learning_phase(): 1})

                if b%100 == 0:
                    acc = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:x_batch, self.labels:y_batch,
                                                     K.learning_phase(): 1})
                    print('training epoch : {}, batch : {}, accuracy : {}'.format(e, b, acc))

            self.valid()
            self.model.save_weights('./weights_{}.h5'.format(e))

    def valid(self, weights_file = None):
        if weights_file is not None:
            self.model.load_weights(weights_file)

        val_idx = np.random.randint(self.x_test.shape[0], size = 128)
        x_val = self.x_test[val_idx]
        y_val = self.y_test[val_idx]
        acc_val = self.sess.run(self.accuracy,
                                feed_dict = {self.images:x_val, self.labels:y_val,
                                             K.learning_phase(): 1})
        print('validation accuracy : {}'.format(acc_val))

    def close(self):
        self.sess.close()
    

def main(batch_size, num_epochs, weights_file):
    
    trainer = TrainCNN()
    if weights_file is None:
        trainer.train(batch_size, num_epochs)
    else:
        trainer.valid(weights_file)

    trainer.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type = int, default = 128,
                        help = 'batch size for train, default [128]')
    parser.add_argument('-e', '--num_epochs', type = int, default = 20,
                        help = 'num epochs, default [20]')
    parser.add_argument('-w', '--weights_file', type = str, default = None,
                        help = 'weight file path, if require validation')
    args = parser.parse_args()

    main(**vars(args))
