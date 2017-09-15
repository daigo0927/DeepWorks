# coding:utf-8

import numpy as np
import tensorflow as tf
sess = tf.Session()

import argparse

from keras import backend as K
K.set_session(sess)
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

def main():

    # chain construction
    images = tf.placeholder(tf.float32, (None, 32, 32, 3), name = 'images')
    labels = tf.placeholder(tf.float32, (None, 10), name = 'labels')
    model = build_simpleCNN()
    labels_ = model(images)
    loss = tf.reduce_mean(categorical_crossentropy(labels, labels_))
    accuracy = tf.reduce_mean(tf.reduce_sum(labels*labels_, axis = 1))
    
    opt = tf.train.AdamOptimizer().minimize(loss, var_list = model.trainable_weights)
    
    # get cifar10 data
    print('load cifar10 data ...')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    data_size = x_train.shape[0]
    batch_size = 128
    num_batches = int(data_size/batch_size)
    num_epochs = 20

    sess.run(tf.global_variables_initializer())

    for e in np.arange(num_epochs):
        permute_idx = np.random.permutation(np.arange(data_size))
        
        for b in np.arange(num_batches):

            # データとラベルのサンプル
            x_batch = x_train[permute_idx[b*batch_size:(b+1)*batch_size]]
            y_batch = y_train[permute_idx[b*batch_size:(b+1)*batch_size]]

            sess.run(opt, feed_dict = {images:x_batch, labels:y_batch,
                                       K.learning_phase(): 1})

            if b%100 == 0:
                acc = sess.run(accuracy, feed_dict = {images:x_batch, labels:y_batch,
                                                      K.learning_phase(): 1})
                print('training epoch : {}, batch : {}, accuracy : {}'.format(e, b, acc))

        val_idx = np.random.randint(x_test.shape[0], size = 128)
        x_val = x_test[val_idx]
        y_val = y_test[val_idx]
        acc_val = sess.run(accuracy, feed_dict = {images:x_val, labels:y_val,
                                                  K.learning_phase(): 1})
        print('validation epoch : {}, accuracy : {}'.format(e, b, acc_val))
        
        model.save_weights('./weights_{}.h5'.format(e))

def valid(weights_file):
    
    images = tf.placeholder(tf.float32, (None, 32, 32, 3), name = 'images')
    labels = tf.placeholder(tf.float32, (None, 10), name = 'labels')
    model = build_simpleCNN()
    labels_ = model(images)
    loss = tf.reduce_mean(categorical_crossentropy(labels, labels_))
    accuracy = tf.reduce_mean(tf.reduce_sum(labels*labels_, axis = 1))
    
    model.load_weights(weights_file)

    print('load cifar10 data ...')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    val_idx = np.random.randint(x_test.shape[0], size = 128)
    x_val = x_test[val_idx]
    y_val = y_test[val_idx]
    acc = sess.run(accuracy, feed_dict = {images:x_val, labels:y_val,
                                          K.learning_phase(): 1})
    print('validation accuracy : {}'.format(acc))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_file', type = str, default = None,
                        help = 'weight file path, if require validation')
    args = parser.parse_args()

    if args.weights_file is None:
        main()
    else:
        valid(args.weights_file)

    sess.close()
