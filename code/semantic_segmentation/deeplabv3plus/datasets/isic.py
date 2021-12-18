import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import random

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

def isic_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class ISIC(data.Dataset):
    cmap = isic_cmap()
    def __init__(self, root, image_set='train', augmentation_prob = 0.2, transform=None):
        self.root = root
        self.GT_paths = root[:-1]+'_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_set = image_set
        self.transform = transform
        self.augmentation_prob = augmentation_prob
        self.RotationDegree = [0,90,180,270]
        print("# of {} samples: {}".format(self.image_set, len(self.image_paths)))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]
        GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'

        img = Image.open(image_path).convert('RGB')
        target = Image.open(GT_path)
        target = Image.fromarray(np.asarray(target)/255)


        if self.transform is not None:
            img, target = self.transform(img, target)


        return img, target


    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def decode_target(cls, target):
        """decode semantic mask to RGB image"""
        return cls.cmap[target]

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
