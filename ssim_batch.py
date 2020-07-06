"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
# from UNIT_utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
#from UNIT_trainer import MUNIT_Trainer, UNIT_Trainer
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from trainer import ERGAN_Trainer
import argparse
from torch.autograd import Variable
from data import ImageFolder
import torchvision.utils as vutils
import sys
import torch
import os, random
import numpy as np
from torchvision import transforms
from PIL import Image
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--A', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--B', type=str, help="style image path")
parser.add_argument('--a2b', action='store_true', help="for a2b")
parser.add_argument('--seed', type=int, default=10, help="random seed")
# parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
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

# imagea_names = ImageFolder(opts.A, transform=None, return_paths=True)
# imageb_names = ImageFolder(opts.B, transform=None, return_paths=True)

data_loader_a = get_data_loader_folder(opts.A, 1, False, new_size=new_size, height=224, width=224,crop=False)
data_loader_b = get_data_loader_folder(opts.B, 1, False, new_size=new_size, height=224, width=224,crop=False)

# Setup model and data loader
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

if 'new_size' in config:
    new_size = config['new_size']

def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp


def img_transform(img):
    transform = transforms.Compose([# transforms.CenterCrop((120, 120)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = Variable(transform(Image.open(img).convert('RGB')).unsqueeze(0).cuda())
    return image

with torch.no_grad():

     a = 100
     b = 100
    # Start testing
     if opts.trainer == 'ERGAN':

         dir_a = os.listdir(opts.A)
         dir_b = os.listdir(opts.B)
         if opts.a2b:
             sample_a = random.sample(dir_a, a)
             sample_b = random.sample(dir_b, b)
             i = 0
             for a in sample_a:
                #print(a)
                images_a = img_transform( os.path.join(opts.A ,a))
                c_a, _ = trainer.gen_a.encode(images_a)
                j = 0
                for b in sample_b:

                    images_b = img_transform(os.path.join(opts.B , b))
                    _, s_b_fake = trainer.gen_b.encode(images_b)
                    for n in range(images_b.size(0)):
                        s_b = s_b_fake[n].unsqueeze(0)
                        # s_b[s_b > 0.7] = 0.7
                        # s_b[s_b < -0.7] = -0.7
                        outputs = trainer.gen_b.decode(c_a, s_b, images_a)
                        im = recover(outputs[0].data.cpu())
                        im = Image.fromarray(im.astype('uint8'))
                        # path = os.path.join(opts.output_folder, os.path.basename(b))
                        path = os.path.join(opts.output_folder+"%03s"%i, '{:06d}.jpg'.format(j))
                        if not os.path.exists(os.path.dirname(path)):
                            os.makedirs(os.path.dirname(path))
                        im = im.resize((120, 120), Image.ANTIALIAS)
                        im.save(path)
                    j = j + 1
                i = i+1

         else:
             sample_b = random.sample(dir_b, a)
             sample_a = random.sample(dir_a, b)
             i = 0
             for b in sample_b:
                 # print(a)
                 images_b = img_transform(os.path.join(opts.B, b))
                 c_b, _ = trainer.gen_b.encode(images_b)
                 j = 0
                 for a in sample_a:

                     images_a = img_transform(os.path.join(opts.A, a))
                     _, s_a_fake = trainer.gen_a.encode(images_a)
                     for n in range(images_a.size(0)):
                         s_a = s_a_fake[n].unsqueeze(0)
                         # s_a[s_a > 0.7] = 0.7
                         # s_a[s_a < - 0.7] = -0.7
                         outputs = trainer.gen_a.decode(c_b, s_a, images_b)
                         im = recover(outputs[0].data.cpu())
                         im = Image.fromarray(im.astype('uint8'))
                         # path = os.path.join(opts.output_folder, os.path.basename(a))
                         path = os.path.join(opts.output_folder + "%03s" % i, '{:06d}.jpg'.format(j))
                         if not os.path.exists(os.path.dirname(path)):
                             os.makedirs(os.path.dirname(path))
                         im = im.resize((120, 120), Image.ANTIALIAS)
                         im.save(path)
                     j = j + 1
                 i = i + 1








