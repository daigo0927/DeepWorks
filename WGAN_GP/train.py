# coding:utf-8

import os, sys
import numpy as np
import pdb
from PIL import Image
import argparse

import tensorflow as tf
import sonnet as snt

from model import GeneratorDeconv, Discriminator

sys.path.append(os.pardir)
from misc.utils import combine_images
from misc.dataIO import InputSampler

    
class WassersteinGAN(object):

    def __init__(self,
                 z_dim, image_size,
                 lr_d, lr_g):

        self.sess = tf.Session()

        self.z_dim = z_dim
        self.image_size = image_size

        self.gen = GeneratorDeconv(input_size = z_dim,
                                   image_size = image_size)
        self.disc = Discriminator()

        self._build_graph(lr_d = lr_d, lr_g = lr_g)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self, lr_d, lr_g):

        self.z = tf.placeholder(tf.float32, (None, self.z_dim))
        self.x_ = self.gen(self.z) # fake image
        
        self.x = tf.placeholder(tf.float32, # true image
                                (None, self.image_size, self.image_size, 3))
        
        self.d_ = self.disc(self.x_, reuse = False)
        self.d = self.disc(self.x)

        # Wasserstein based loss
        self.g_loss = -tf.reduce_mean(self.d_)
        self.d_loss = -(tf.reduce_mean(self.d) - tf.reduce_mean(self.d_))

        # gradient penalty
        alpha = tf.random_uniform(shape = tf.shape(self.x),
                                  minval = 0., maxval = 1.,)
        differ = self.x_ - self.x
        interp = self.x + (alpha * differ)
        grads = tf.gradients(self.disc(interp), [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads),
                                       reduction_indices = [3]))
        grad_penalty = tf.reduce_mean((slopes - 1.)**2)
        self.d_loss += 10 * grad_penalty

        self.lr_d = lr_d
        self.lr_g = lr_g

        self.d_opt = tf.train.AdamOptimizer(learning_rate = self.lr_d,
                                            beta1 = 0., beta2 = 0.9)\
                     .minimize(self.d_loss, var_list = self.disc.vars)
        self.g_opt = tf.train.AdamOptimizer(learning_rate = self.lr_g,
                                            beta1 = 0., beta2 = 0.9)\
                     .minimize(self.g_loss, var_list = self.gen.vars)
        
    def train(self,
              nd,
              sampler, # input data sampler, defined in misc.dataIO.py
              epochs, 
              batch_size, 
              sampledir, modeldir): # result (image/model) saving dir
        
        num_batches = int(sampler.data_size/batch_size)
        print('epochs : {}, number of batches : {}'.format(epochs, num_batches))

        # training iteration
        for e in range(epochs):
                
            for batch in range(num_batches):

                if batch in np.linspace(0, num_batches, sampler.split+1, dtype = int):
                    sampler.reload()

                d_iter = nd                    
                
                for _ in range(d_iter):
                    bx = sampler.image_sample(batch_size)
                    bz = sampler.noise_sample(batch_size)
                    self.sess.run(self.d_opt, feed_dict = {self.x: bx, self.z: bz})

                bz = sampler.noise_sample(batch_size, self.z_dim)
                self.sess.run(self.g_opt, feed_dict = {self.z: bz})

                if batch%10 == 0:
                    d_loss, g_loss = self.sess.run([self.d_loss, self.g_loss],
                                                   feed_dict = {self.x: bx, self.z: bz})
                    print('epoch : {}, batch : {}, d_loss : {}, g_loss : {}'\
                          .format(e, batch, d_loss, g_loss))

                if batch%100 == 0:
                    fake_seed = sampler.noise_sample(9, self.z_dim)
                    fake_sample = self.sess.run(self.x_,
                                                feed_dict = {self.z: fake_seed})
                    fake_sample = combine_images(fake_sample)
                    fake_sample = fake_sample*127.5 + 127.5
                    Image.fromarray(fake_sample.astype(np.uint8))\
                         .save(sampledir + '/sample_{}_{}.png'.format(e, batch))

            self.saver.save(self.sess, modeldir + '/model{}.ckpt'.format(e))

            
def main(epochs,
         lr_g, lr_d,
         train_size, batch_size, nd,
         target_size, image_size,
         datadir,
         split, loadweight, modeldir, sampledir):

    sampler = InputSampler(datadir = datadir,
                           target_size = target_size, image_size = image_size,
                           split = split, num_utilize = train_size)

    wgan = WassersteinGAN(z_dim = 100, image_size = image_size,
                          lr_d =  lr_d,
                          lr_g =  lr_g)
    
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    if not os.path.exists(sampledir):
        os.mkdir(sampledir)

    wgan.train(nd = nd,
               sampler = sampler,
               epochs = epochs,
               batch_size = batch_size,
               sampledir = sampledir,
               modeldir = modeldir)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # optimization
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help = 'number of epochs [20]')
    parser.add_argument('--lr_g', type = float, default = 1e-4,
                        help = 'learning rate for generator [1e-4]')
    parser.add_argument('--lr_d', type = float, default = 1e-4,
                        help = 'learning rate for discriminator [1e-4]')
    parser.add_argument('--train_size', type = int, default = np.inf,
                        help = 'size of trainind data [np.inf]')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help = 'size of mini-batch [64]')
    parser.add_argument('--nd', type = int, default = 5,
                        help = 'training schedule for dicriminator by generator [5]')
    # data I/O
    parser.add_argument('--target_size', type = int, default = 108,
                        help = 'target area of training data [108]')
    parser.add_argument('--image_size', type = int, default = 64,
                        help = 'size of generated image [64]')
    parser.add_argument('-d', '--datadir', type = str, nargs = '+', required = True,
                        help = 'path to dir contains training (image) data')
    parser.add_argument('--split', type = int, default = 5,
                        help = 'load data, by [5] split')
    parser.add_argument('--loadweight', type = str, default = False,
                        help = 'path to dir conrtains trained weights [False]')
    parser.add_argument('--modeldir', type = str, default = './model',
                        help = 'path to dir put trained weighted [./model]')
    parser.add_argument('--sampledir', type = str, default = './image',
                        help = 'path to dir put generated image samples [./image]')
    args = parser.parse_args()

    for key in vars(args).keys():
        print('{} : {}'.format(key, vars(args)[key]))
    
    main(**vars(args))
