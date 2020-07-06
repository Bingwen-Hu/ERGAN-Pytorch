"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from UNIT_utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from UNIT_trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
from data import ImageFolder
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--A', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--B', type=str, default='', help="style image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', action='store_true', help="for a2b , else for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

imagea_names = ImageFolder(opts.A, transform=None, return_paths=True)
imageb_names = ImageFolder(opts.B, transform=None, return_paths=True)
data_loader_a = get_data_loader_folder(opts.A, 1, False, new_size=config['new_size'], height=224,
                                     width=224,crop=False)
data_loader_b = get_data_loader_folder(opts.B, 1, False, new_size=config['new_size'], height=224,
                                     width=224,crop=False)

# Setup model and data loader
config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

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

encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function
style_decode = trainer.gen_a.decode if opts.a2b else trainer.gen_b.decode # decode function

def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp

with torch.no_grad():

    if opts.trainer == 'MUNIT':

        for i, (images_a, images_b, imagea_names, imageb_names) in enumerate(
                zip(data_loader_a, data_loader_b, imagea_names, imageb_names)):

            images_a = images_a.cuda()
            images_b = images_b.cuda()
            basename_b = os.path.basename(imageb_names[1])
            basename_a = os.path.basename(imagea_names[1])

            c_a, s_a_fake = trainer.gen_a.encode(images_a)
            c_b, s_b_fake = trainer.gen_b.encode(images_b)

            if opts.a2b:
                for j in range(images_b.size(0)):
                    s_b = s_b_fake[j].unsqueeze(0)

                    outputs = trainer.gen_b.decode(c_a, s_b)
                    im = recover(outputs[0].data.cpu())
                    im = Image.fromarray(im.astype('uint8'))
                    path = os.path.join(opts.output_folder, basename_a)
                    im = im.resize((120, 120), Image.ANTIALIAS)
                    im.save(path)
                    # path = os.path.join(opts.output_folder, '{:06d}.jpg'.format(i))
                    # vutils.save_image(outputs.data, path, padding=0, normalize=True)

            else:
                for j in range(images_a.size(0)):
                    s_a = s_a_fake[j].unsqueeze(0)
                    outputs = trainer.gen_a.decode(c_b, s_a)
                    im = recover(outputs[0].data.cpu())
                    im = Image.fromarray(im.astype('uint8'))
                    path = os.path.join(opts.output_folder, basename_b)
                    im = im.resize((120, 120), Image.ANTIALIAS)
                    im.save(path)
                    # path = os.path.join(opts.output_folder, '{:06d}.jpg'.format(i))
                    # vutils.save_image(outputs.data, path, padding=0, normalize=True)

    elif opts.trainer == 'UNIT':
        for i, (images_a, images_b, imagea_names, imageb_names) in enumerate(
                zip(data_loader_a, data_loader_b, imagea_names, imageb_names)):

            images_a = images_a.cuda()
            images_b = images_b.cuda()

            basename_b = os.path.basename(imageb_names[1])
            basename_a = os.path.basename(imagea_names[1])

            if opts.a2b:
                h_a, n_a = encode(images_a)
                outputs = decode(h_a + n_a)
                im = recover(outputs[0].data.cpu())
                im = Image.fromarray(im.astype('uint8'))
                path = os.path.join(opts.output_folder, basename_a)
                im = im.resize((120, 120), Image.ANTIALIAS)
                im.save(path)
            else:
                h_b, n_b = encode(images_b)
                outputs = decode(h_b + n_b)
                im = recover(outputs[0].data.cpu())
                im = Image.fromarray(im.astype('uint8'))
                path = os.path.join(opts.output_folder, basename_b)
                im = im.resize((120, 120), Image.ANTIALIAS)
                im.save(path)
            # path = os.path.join(opts.output_folder, '{:06d}.jpg'.format(i))
            # vutils.save_image(outputs.data, path, padding=0, normalize=True)
            # if not opts.output_only:
            #      # also save input images
            #      vutils.save_image(images_b.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)),
            #                        padding=0, normalize=True)
    else:
         pass



