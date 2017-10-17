# coding:utf-8

import os, sys
sys.path.append(os.pardir)

import numpy as np
import tensorflow as tf
import sonnet as snt

import argparse
from tqdm import tqdm

from misc.utils import load_cifar10

def build_simpleCNN(num_output = 10):

    def build(inputs):
        x = snt.Conv2D(output_channels = 64, kernel_shape = (4, 4),
                       stride = (1, 1), padding = 'VALID')(inputs)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x,
                           ksize = [1, 2, 2, 1],
                           strides = [1, 2, 2, 1],
                           padding = 'SAME')
        x = snt.Conv2D(output_channels = 128, kernel_shape = (4, 4),
                       stride = (1, 1), padding = 'VALID')(x)
        x = snt.BatchNorm()(x, is_training = True)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x,
                           ksize = [1, 2, 2, 1],
                           strides = [1, 2, 2, 1],
                           padding = 'SAME')
        x = snt.BatchFlatten()(x)
        outputs = snt.Linear(output_size = num_output)(x)
        return outputs
    
    return snt.Module(build)

class TrainCNN(object):

    def __init__(self):

        self.sess = tf.Session()
                
        self._build_graph()
        self._load_data()
    
    def _build_graph(self):
        
        self.images = tf.placeholder(tf.float32, (None, 32, 32, 3), name = 'images')
        self.labels = tf.placeholder(tf.float32, (None, 10), name = 'labels')
        self.model = build_simpleCNN()
        
        self.logits = self.model(self.images)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels = self.labels, logits = self.logits))
        
        self.preds = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.reduce_sum(self.labels*self.preds, axis = 1))

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

        self.saver = tf.train.Saver()

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
                              feed_dict = {self.images:x_batch, self.labels:y_batch})

                if b%100 == 0:
                    acc = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:x_batch, self.labels:y_batch})
                    print('training epoch : {}, batch : {}, accuracy : {}'.format(e, b, acc))

            self.valid()
            self.saver.save(self.sess, './model_{}.ckpt'.format(e))

    def valid(self, batch_size = 128, weights_file = None):
        
        if weights_file is not None:
            self.saver.restore(self.sess, weights_file)

        data_size = self.x_test.shape[0]
        num_batches = int(data_size/batch_size)

        acc_vals = []
        permute_idx = np.random.permutation(np.arange(data_size))
        for b in tqdm(np.arange(num_batches)):
            x_val = self.x_test[permute_idx[b*batch_size:(b+1)*batch_size]]
            y_val = self.y_test[permute_idx[b*batch_size:(b+1)*batch_size]]

            acc_val = self.sess.run(self.accuracy,
                                    feed_dict = {self.images:x_val, self.labels:y_val})
            acc_vals.append(acc_val)
            
        print('validation accuracy : {}'.format(np.mean(acc_vals)))

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
