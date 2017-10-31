# coding:utf-8

import numpy as np
from PIL import Image
import argparse

import tensorflow as tf
from model import U_Net

import os, sys
sys.path.append(os.pardir)
from misc.utils import combine_images
from misc.dataIO import InputSampler

class Trainer(object):

    def __init__(self,
                 image_size):

        self.sess = tf.Session()
        self._build_graph(image_size = image_size)

    def _build_graph(self, image_size):

        self.image_size = image_size
        self.images = tf.placeholder(tf.float32,
                                     shape = (None, image_size, image_size, 3))
        images_mini = tf.image.resize_images(self.images,
                                             size = (int(image_size/4),
                                                     int(image_size/4)))
        self.images_blur = tf.image.resize_images(images_mini,
                                                  size = (image_size, image_size))
        
        self.net = U_Net(output_ch = 3)
        self.images_reconst = self.net(self.images_blur, reuse = False)
        # self.image_reconst can be [-inf +inf], so need to clip its value if visualize them as images.
        self.loss = tf.reduce_mean((self.images_reconst - self.images)**2)
        self.opt = tf.train.AdamOptimizer()\
                           .minimize(self.loss, var_list = self.net.vars)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def train(self,
              sampler,
              num_epochs, batch_size,
              sampledir, modeldir):

        num_batches = int(sampler.data_size/batch_size)
        print('epochs : {}, number of batches : {}'\
              .format(num_epochs, num_batches))

        # training iteration
        for e in range(num_epochs):
            for batch in range(num_batches):

                if batch in np.linspace(0, num_batches, sampler.split+1, dtype = int):
                    sampler.reload()

                imgs = sampler.image_sample(batch_size)
                self.sess.run(self.opt, feed_dict = {self.images:imgs})

                if batch%10 == 0:
                    loss = self.sess.run(self.loss, feed_dict = {self.images:imgs})
                    print('epoch : {}, batch : {}, loss : {}'\
                          .format(e, batch, loss))

                if batch%100 == 0:
                    imgs_ = sampler.image_sample(9)
                    imgs_blur, imgs_reconst \
                        = self.sess.run([self.images_blur, self.images_reconst],
                                        feed_dict = {self.images:imgs_})
                    imgs_blur = combine_images(imgs_blur)*127.5 + 127.5
                    # clip imgs_reconst value [-1, 1] for visualize
                    imgs_reconst = np.clip(imgs_reconst, -1., 1.)
                    imgs_reconst = combine_images(imgs_reconst)*127.5 + 127.5
                    Image.fromarray(imgs_blur.astype(np.uint8))\
                         .save(sampledir + '/blur_{}_{}.png'.format(e, batch))
                    Image.fromarray(imgs_reconst.astype(np.uint8))\
                         .save(sampledir + '/reconst_{}_{}.png'.format(e, batch))

            self.saver.save(self.sess, modeldir + '/model{}.ckpt'.format(e))

def main(epochs,
         train_size, batch_size,
         target_size, image_size,
         datadir,
         split, modeldir, sampledir):

    sampler = InputSampler(datadir = datadir,
                           target_size = target_size, image_size = image_size,
                           split = split, num_utilize = train_size)

    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    if not os.path.exists(sampledir):
        os.mkdir(sampledir)

    trainer = Trainer(image_size = image_size)
    trainer.train(sampler = sampler,
                  num_epochs = epochs,
                  batch_size = batch_size,
                  sampledir = sampledir,
                  modeldir = modeldir)

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
    parser.add_argument('--target_size', type = int, default = 200,
                        help = 'target cropping area for original images [200]')
    parser.add_argument('--image_size', type = int, default = 128,
                        help = 'size of dealing images [128]')
    parser.add_argument('-d', '--datadir', type = str, nargs = '+', required = True,
                        help = '/path/to/dir contains training data')
    parser.add_argument('--split', type = int, default = 5,
                        help = 'load data, by [5] split')
    parser.add_argument('-m', '--modeldir', type = str, default = './model',
                        help = '/path/to/dir put trained model weights [./model]')
    parser.add_argument('-s', '--sampledir', type = str, default = './image',
                        help = '/path/to/dir put sample trained images [./image]')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{} : {}'.format(key, value))

    main(**vars(args))
