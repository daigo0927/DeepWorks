# coding:utf-8

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import pdb

from .utils import *

class InputSampler:

    def __init__(self,
                 datadir, # path/to/dir contains image files
                 labelfile = None, # path/to/labelfile if needed
                 target_size = 108, # crop size for target images
                 image_size = 64, # scaled image size
                 split = 5, # value for split whole data and load on memory
                 num_utilize = np.inf): # utilize size of while data
        
        self.datadir = datadir
        self.target_size = target_size
        self.image_size = image_size
        self.split = split

        self.image_paths = []
        for d in self.datadir:
            self.image_paths += glob(d + '/*.jpg')
        self.data_size = min(len(self.image_paths), num_utilize)
        print('data size : {}'.format(self.data_size))

        permute_idx = np.random.permutation(self.data_size)
        self.image_paths = np.array(self.image_paths)
        self.image_paths = self.image_paths[permute_idx]

        self.label = None
        if labelfile is not None:
            
            if 'celeba' in labelfile:
                self.label = LabelfileLoader.load_celebA(labelfile)
            else:
                self.label = LabelfileLoader.load(labelfile)

            assert len(self.label) == self.data_size, 'data and label have valid size.'
            print('label names : {}'.format(self.label.columns))
            self.label = self.label.ix[permute_idx]

        self.data = None # splitted data variable
        self.split_label = None # splitted label variable

    def load(self):
        self.reload()

    def reload(self):
        split_idx = np.random.choice(np.arange(self.data_size),
                                     int(self.data_size/self.split),
                                     replace = False)
        split_paths = self.image_paths[split_idx]
        
        print('split data loading ...')
        self.data = np.array([get_image(p, self.target_size, self.image_size)
                              for p in tqdm(split_paths)])

        if self.label is not None:
            self.split_label = self.label.ix[split_idx]

    def image_sample(self, batch_size):
        images = self.data[np.random.choice(len(self.data),
                                            batch_size,
                                            replace = False)]
        return images

    def image_label_sample(self, batch_size):
        sample_idx = np.random.choice(len(self.data),
                                      batch_size,
                                      replace = False)
        images = self.data[sample_idx]
        labels = self.split_label.ix[sample_idx]
        
        return images, labels

    def noise_sample(self, batch_size, noise_dim = 100):
        noise = np.random.uniform(-1, 1, (batch_size, noise_dim))
        return noise

    
class LabelfileLoader(object):

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            label = pd.read_csv(f, header = None)
        return label

    @staticmethod
    def load_celebA(filepath):
        with open(filepath, 'rb') as f:
            label = pd.read_csv(f, delim_whitespace=True, header=1)
        return label

