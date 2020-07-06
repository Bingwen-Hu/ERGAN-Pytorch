"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import sys
sys.path.append('.')
from utils import get_config
from trainer import ERGAN_Trainer, to_gray
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import imageio
import os
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

name = 'eye'

if not os.path.isdir('models/%s'%name):
    assert 0, "please change the name to your model name"

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default="./", help="output image path")
parser.add_argument('--input_folder', type=str, default="inputs/swap/", help="input image path")
parser.add_argument('--config', type=str, default='./outputs/%s/config.yaml'%name, help="net configuration")
parser.add_argument('--checkpoint_gen', type=str, default="./outputs/%s/checkpoints/gen_00100000.pt"%name, help="checkpoint of autoencoders")
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--trainer', type=str, default='ERGAN', help="ERGAN")


opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1

# Setup model and data loader
if opts.trainer == 'ERGAN':
    trainer = ERGAN_Trainer(config)
else:
    sys.exit("Only support ERGAN")

state_dict_gen = torch.load(opts.checkpoint_gen)
trainer.gen_a.load_state_dict(state_dict_gen['a'], strict=False)
trainer.gen_b.load_state_dict(state_dict_gen['b'], strict=False)

trainer.cuda()
trainer.eval()
encode = trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode # encode function
decode = trainer.gen_b.decode # decode function
data_transforms = transforms.Compose([
        transforms.Resize((224,224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_datasets1 = datasets.ImageFolder(opts.input_folder+'/1', data_transforms)
image_datasets2 = datasets.ImageFolder(opts.input_folder+'/2', data_transforms)
dataloader_content = torch.utils.data.DataLoader(image_datasets1, batch_size=2, shuffle=False, num_workers=1)
dataloader_style = torch.utils.data.DataLoader(image_datasets2, batch_size=2, shuffle=False, num_workers=1)
######################################################################
# recover image
# -----------------
def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp

im = {}
# data2 = next(iter(dataloader_structure))
# bg_img, _ = data2
# bg_img = Variable(bg_img.cuda())
ff = []
gif = []
with torch.no_grad():
    for data in dataloader_content:
        id_img, _ = data
        id_img = id_img.cuda()
        n, c, h, w = id_img.size()
        f, _ = encode(id_img)
        for data2 in dataloader_style:
            bg_img, _ = data2
            bg_img = bg_img.cuda()
            _, s = style_encode(bg_img)
        # Start testing
            for count in range(2):
                input1 = recover(id_img[count].squeeze().data.cpu())
                im[count] = input1

                for i in range(10):
                    f_tmp = f[count,:,:,:]
                    tmp_s = 0.1*i*s[0] + (1-0.1*i)*s[1]
                    tmp_s = tmp_s.view(1, -1).contiguous()
                    outputs = decode(f_tmp.unsqueeze(0), tmp_s, bg_img)
                    tmp = recover(outputs[0].data.cpu())
                    im[count] = np.concatenate((im[count], tmp), axis=1)
                    #gif.append(tmp)
        break

# save long image
pic = np.concatenate( (im[0], im[1]) , axis=0)
pic = Image.fromarray(pic.astype('uint8'))
pic.save('smooth.jpg')

# save gif
#imageio.mimsave('./smooth.gif', gif)