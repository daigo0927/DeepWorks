# coding:utf-8

import numpy as np
import argparse
import time

import tensorflow as tf

import os, sys
sys.path.append(os.pardir)
from ResNet_tf.model import ResNetBuilder
from Sonnet.train import build_simpleCNN
from misc.utils import load_cifar10, load_cifar100

class Trainer(object):

    def __init__(self):

        self.sess = tf.Session()
        self._load_cifar10()
        self._build_graph()

    def _load_cifar10(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10()

    def _build_graph(self):

        self.images = tf.placeholder(tf.float32,
                                     shape = (None, 32, 32, 3), name = 'images')
        self.labels = tf.placeholder(tf.float32,
                                     shape = (None, 10), name = 'labels')

        self.net = build_simpleCNN()
        # self.net = ResNetBuilder.build_resnet18(num_output = 100)
        self.logits = self.net(self.images)#, reuse = False)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels = self.labels, logits = self.logits))

        self.preds = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.reduce_sum(self.labels*self.preds, axis = 1))
        
        self.opt = tf.train.AdamOptimizer()\
                           .minimize(self.loss)#, var_list = self.net.vars)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def train(self,
              num_epochs,
              batch_size):

        num_batches = int(len(self.x_train)/batch_size)
        print('epochs : {}, number of batches : {}'\
              .format(num_epochs, num_batches))

        lap_times = []
        # training iteration
        for e in range(num_epochs):
            permute_idx = np.random.permutation(np.arange(50000))
            lap_time = []
            for b in range(num_batches):

                x_batch = self.x_train[permute_idx[b*batch_size:(b+1)*batch_size]]
                y_batch = self.y_train[permute_idx[b*batch_size:(b+1)*batch_size]]

                s_time = time.time()
                self.sess.run(self.opt,
                              feed_dict = {self.images:x_batch, self.labels:y_batch})
                e_time = time.time()
                lap_time.append(e_time - s_time)

                if b%10 == 0:
                    acc = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:x_batch,
                                                     self.labels:y_batch})
                    print('epoch : {}, batch : {}, accuracy : {}'\
                          .format(e, b, acc))

            # record single epoch training lap-time
            lap_times.append(np.mean(lap_time))
            
            # validation
            accs_val = []
            for b in range(int(len(self.x_test)/batch_size)):
                x_val = self.x_test[b*batch_size:(b+1)*batch_size]
                y_val = self.y_test[b*batch_size:(b+1)*batch_size]
                acc_val = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:x_val,
                                                     self.labels:y_val})
                accs_val.append(acc_val)
            print('{} epoch validation accuracy {}'.format(e, np.mean(accs_val)))

            # save trained model
            self.saver.save(self.sess, './model_tf/model{}.ckpt'.format(e))

        # record training time
        with open('./lap_record.csv', 'a') as f:
            f.write('tensorflow')
            for lap in lap_times:
                f.write(',' + str(lap))

def train_tf(epochs, batch_size):

    if not os.path.exists('./model_tf'):
        os.mkdir('./model_tf')

    trainer = Trainer()
    trainer.train(num_epochs = epochs,
                  batch_size = batch_size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # optimization
    parser.add_argument('-e', '--epochs', type = int, default = 20,
                        help = 'number of epochs [20]')
    parser.add_argument('-b', '--batch_size', type = int, default = 64,
                        help = 'size of mini-batch [64]')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{} : {}'.format(key, value))

    train_tf(**vars(args))
