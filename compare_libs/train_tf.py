# coding:utf-8

import numpy as np
import argparse
import time

import tensorflow as tf

import os, sys
sys.path.append(os.pardir)
from ResNet_tf.model import ResNetBuilder
from misc.dataIO import Food101Sampler

class Trainer(object):

    def __init__(self,
                 image_size):

        self.sess = tf.Session()
        self._build_graph(image_size = image_size)

    def _build_graph(self, image_size):

        self.image_size = image_size
        self.images = tf.placeholder(tf.float32,
                                     shape = (None, image_size, image_size, 3))
        self.labels = tf.placeholder(tf.float32,
                                     shape = (None, 101))

        self.net = ResNetBuilder.build_resnet18(num_output = 101)
        self.logits = self.net(self.images, reuse = False)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels = self.labels, logits = self.logits))

        self.preds = tf.nn.softmax(self.logits)
        self.accuracy = tf.reduce_mean(tf.reduce_sum(self.labels*self.preds, axis = 1))
        
        self.opt = tf.train.AdamOptimizer()\
                           .minimize(self.loss, var_list = self.net.vars)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def train(self,
              sampler,
              num_epochs,
              batch_size):

        num_batches = int(sampler.data_size/batch_size)
        print('epochs : {}, number of batches : {}'\
              .format(num_epochs, num_batches))

        # imgs_val, labs_val = sampler.load_testset()
        # num_valbatches = int(len(imgs_val)/batch_size)

        lap_times = []
        # training iteration
        for e in range(num_epochs):
            lap_time = []
            for batch in range(num_batches):

                if batch in np.linspace(0, num_batches, sampler.split+1, dtype = int):
                    sampler.reload()

                imgs, labs = sampler.sample(batch_size) # value [-1 - 1]
                s_time = time.time()
                self.sess.run(self.opt,
                              feed_dict = {self.images:imgs, self.labels:labs})
                e_time = time.time()
                lap_time.append(e_time - s_time)

                if batch%10 == 0:
                    acc = self.sess.run(self.accuracy,
                                        feed_dict = {self.images:imgs,
                                                     self.labels:labs})
                    print('epoch : {}, batch : {}, accuracy : {}%'\
                          .format(e, batch, acc))

            # record single epoch training lap-time
            lap_times.append(np.mean(lap_time))
            
            # validation
            for v_batch in range(num_valbatches):
                val_accs = []
                img_val = imgs_val[v_batch*batch_size:(v_batch+1)*batch_size]
                lab_val = labs_val[v_batch*batch_size:(v_batch+1)*batch_size]
                v_acc = self.sess.run(self.accuracy,
                                      feed_dict = {self.images:img_val,
                                                   self.labels:lab_val})
                val_accs.append(v_acc)
            print('{} epochs validation accuracy {}%'.format(e, np.mean(val_accs)))

            # save trained model
            self.saver.save(self.sess, './model_tf/model{}.ckpt'.format(e))

        # record training time
        with open('./lap_record.csv', 'a') as f:
            f.write('tensorflow')
            for lap in lap_times:
                f.write(',' + str(lap))

def train_tf(epochs,
             train_size, batch_size,
             target_size, image_size,
             metadir,
             split):

    sampler = Food101Sampler(metadir = metadir,
                             target_size = target_size, image_size = image_size,
                             split = split, num_utilize = train_size)

    if not os.path.exists('./model_tf'):
        os.mkdir('./model_tf')

    trainer = Trainer(image_size = image_size)
    trainer.train(sampler = sampler,
                  num_epochs = epochs,
                  batch_size = batch_size)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # optimization
    parser.add_argument('-e', '--epochs', type = int, default = 20,
                        help = 'number of epochs [20]')
    parser.add_argument('-t', '--train_size', type = int, default = np.inf,
                        help  = 'size of utilizing data [np.inf(all)]')
    parser.add_argument('-b', '--batch_size', type = int, default = 64,
                        help = 'size of mini-batch [64]')
    # data I/O
    parser.add_argument('--target_size', type = int, default = 512,
                        help = 'target cropping area for original images [512]')
    parser.add_argument('--image_size', type = int, default = 128,
                        help = 'size of dealing images [128]')
    parser.add_argument('-m', '--metadir', type = str, required = True,
                        help = '/path/to/food-101/mera contains food101 metadata')
    parser.add_argument('-s', '--split', type = int, default = 5,
                        help = 'load data, by [5] split')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{} : {}'.format(key, value))

    train_tf(**vars(args))
