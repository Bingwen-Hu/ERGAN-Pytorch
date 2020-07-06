"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
from trainer import ERGAN_Trainer, UNIT_Trainer
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
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celebA_folder', help='Path to the config file.')
parser.add_argument('--A', type=str, default = 'datasets/celebA/trainA_test', help="input image folder A")
parser.add_argument('--B', type=str, default = 'datasets/celebA/trainB',help="input image folder B")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', action='store_true', help="a2b / b2a" )
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='ERGAN', help="ERGAN|UNIT")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']
new_size = config['new_size']
crop_image_height =config['crop_image_height']
crop_image_width =config['crop_image_width']

# Setup model and data loader

data_loader_a = get_data_loader_folder(opts.A, 1, False, new_size=new_size, height=crop_image_height, width=crop_image_width,crop=True)
data_loader_b = get_data_loader_folder(opts.B, 1, False, new_size=new_size, height=crop_image_height, width=crop_image_width,crop=True)
imagea_names = ImageFolder(opts.A, transform=None, return_paths=True)
imageb_names = ImageFolder(opts.B, transform=None, return_paths=True)

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'ERGAN':
    style_dim = config['gen']['style_dim']
    trainer = ERGAN_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support ERGAN|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()

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

# Start testing
with torch.no_grad():
     if opts.trainer == 'ERGAN':

        for i, (images_a, images_b, imagea_names, imageb_names) in enumerate(zip(data_loader_a, data_loader_b, imagea_names, imageb_names)):
            basename_b = os.path.basename(imageb_names[1])
            basename_a = os.path.basename(imagea_names[1])
            #images_a_flip, images_b_flip = flip_lr(images_a).cuda(), flip_lr(images_b).cuda()
            images_a, images_b = images_a.cuda(), images_b.cuda()

            c_a, s_a_fake = trainer.gen_a.encode(images_a)
            # _, s_a_fake_flip = trainer.gen_a.encode(images_a_flip)
            # s_a_fake = (s_a_fake+s_a_fake_flip)/2
            c_b, s_b_fake = trainer.gen_b.encode(images_b)

            if opts.a2b:
                for j in range(images_b.size(0)):
                    s_b = s_b_fake[j].unsqueeze(0)
                    #s_b = style_b[j].unsqueeze(0)
                    # s_b[s_b>0.7]=0.7
                    # s_b[s_b < -0.7] = -0.7
                    outputs = trainer.gen_b.decode(c_a, s_b, images_a)
                    im = recover(outputs[0].data.cpu())
                    im = Image.fromarray(im.astype('uint8'))
                    path = os.path.join(opts.output_folder, basename_a)
                    im = im.resize((new_size, new_size), Image.ANTIALIAS)
                    im.save(path)

            else:
                for j in range(images_a.size(0)):
                    s_a = s_a_fake[j].unsqueeze(0)
                    #  s_a[s_a > 0.7] = 0.7
                    #  s_a[s_a <- 0.7] = -0.7
                    outputs = trainer.gen_a.decode(c_b, s_a, images_b)
                    im = recover(outputs[0].data.cpu())
                    im = Image.fromarray(im.astype('uint8'))
                    path = os.path.join(opts.output_folder, basename_b)
                    im = im.resize((new_size, new_size), Image.ANTIALIAS)
                    im.save(path)
                    #vutils.save_image(images_a.data, os.path.join(opts.output_folder, 'input{:06d}.jpg'.format(i)), padding=0, normalize=True)
     else:
        pass


