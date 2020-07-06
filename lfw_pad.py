"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer import ERGAN_Trainer, UNIT_Trainer
#from network import AdaINGen, MsImageDis, VAEGen
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
from PIL import Image
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('--A', type=str, default = '/home/bingwen/MUNIT-master/datasets/lfw_224_pad', help="input image folder A")
#parser.add_argument('--B', type=str, default = '/home/bingwen/MUNIT-master/datasets/LFW/trainB', help="input image folder B")

opts = parser.parse_args()

output_folder1 = os.path.abspath('/home/bingwen/MUNIT-master/datasets/lfw_224_JPG')
#output_folder2 = os.path.abspath('/home/bingwen/MUNIT-master/datasets/LFW/trainB_pad')

if not os.path.exists(output_folder1):
    os.makedirs(output_folder1)

data_loader_a = get_data_loader_folder(opts.A, 1, False, new_size=224, height=224, width=224,crop=False)
#data_loader_b = get_data_loader_folder(opts.B, 1, False, new_size=224, height=224, width=224,crop=False)
imagea_names = ImageFolder(opts.A, transform=None, return_paths=True)
#imageb_names = ImageFolder(opts.B, transform=None, return_paths=True)

def flip_lr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp

for i, (images_a, imagea_names) in enumerate(zip(data_loader_a, imagea_names)):

            #basename_b = os.path.basename(imageb_names[1])
            basename_a = os.path.basename(imagea_names[1])

            images_a = images_a.cuda()
            #images_b = images_b.cuda()
            images_a = F.pad(images_a, (0, 0, 24, -24), mode='reflect')
            #images_b = F.pad(images_b, (0, 0, 24, -24), mode='reflect')

            (name_a, extention_a) = os.path.splitext(basename_a)
            #(name_b, extention_b) = os.path.splitext(basename_b)
            print(i+1)
            vutils.save_image(images_a.data, os.path.join(output_folder1, name_a +'.jpg'), padding=0, normalize=True)
            #vutils.save_image(images_b.data, os.path.join(output_folder2, name_b +'.png'), padding=0, normalize=True)

else:
    pass


