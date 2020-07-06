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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input_folder', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', action='store_true', help=" a2b ")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=1, help="number of styles to sample")
# parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
#parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
#opts.num_style = 1 if opts.style != '' else opts.num_style

image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 1, False, new_size=config['new_size'], height=config['crop_image_height'],
                                     width=config['crop_image_width'],crop=True)
height = config['crop_image_height']
width = config['crop_image_width']
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
# if opts.a2b:
#     encode = trainer.gen_a.encode
#     style_encode = trainer.gen_b.encode
#     decode = trainer.gen_b.decode
# elif opts.b2a:
#     encode = trainer.gen_b.encode
#     style_encode = trainer.gen_a.encode
#     decode = trainer.gen_a.decode
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
# else:
#     if opts.a2b==1:
#         new_size = config['new_size_a']
#     else:
#         new_size = config['new_size_b']

with torch.no_grad():
     transform = transforms.Compose([transforms.CenterCrop((height, width)),
                                     transforms.Resize((new_size, new_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda())

    # Start testing
     content, _ = encode(style_image)
     if opts.trainer == 'MUNIT':
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            #print(names[1])
            images = Variable(images.cuda(), volatile=True)
            #style = Variable(torch.randn(images.size(0), style_dim, 1, 1).cuda(), volatile=True)
            _, style = style_encode(images)
            for j in range(images.size(0)):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                basename = os.path.basename(names[1])
                path = os.path.join(opts.output_folder, basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
            #
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            # if not opts.output_only:
            #     vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)),
            #                           padding=0, normalize=True)
     elif opts.trainer == 'UNIT':
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            #print(names[1])
            images = Variable(images.cuda(), volatile=True)
            hiddens, _ = encode(style_image)
            #print(hiddens.shape)
            #_, noise = encode(images)
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            outputs = decode(hiddens+noise)
            outputs = (outputs + 1) / 2.
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder, basename)
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            # if not opts.output_only:
            #     # also save input images
            #     vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
     else:
         pass



