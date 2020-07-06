from __future__ import print_function
from UNIT_utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04
from UNIT_trainer import MUNIT_Trainer, UNIT_Trainer
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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--A', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--B', type=str, help="style image path")
parser.add_argument('--a2b', action='store_true', help="for a2b")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

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

if 'new_size' in config:
    new_size = config['new_size']

def img_transform(img):
    transform = transforms.Compose([#transforms.CenterCrop((160, 160)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Variable(transform(Image.open(img).convert('RGB')).unsqueeze(0).cuda())
    return image

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
     dir_a = os.listdir(opts.A)
     dir_b = os.listdir(opts.B)
     sample_a = random.sample(dir_a, 100)
     sample_b = random.sample(dir_b, 100)
    # Start testing
     if opts.trainer == 'MUNIT':

         if opts.a2b:
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
                        outputs = trainer.gen_b.decode(c_a, s_b)
                        im = recover(outputs[0].data.cpu())
                        im = Image.fromarray(im.astype('uint8'))
                        # path = os.path.join(opts.output_folder, os.path.basename(a))
                        path = os.path.join(opts.output_folder + "%03s" % i, '{:06d}.jpg'.format(j))
                        if not os.path.exists(os.path.dirname(path)):
                            os.makedirs(os.path.dirname(path))
                        im = im.resize((120, 120), Image.ANTIALIAS)
                        im.save(path)
                    j = j + 1
                i = i+1

         else:
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
                         outputs = trainer.gen_a.decode(c_b, s_a)
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


     elif opts.trainer == 'UNIT':

         if opts.a2b:
             i = 0
             for a in sample_a:
                 # print(a)
                 images_a = img_transform(os.path.join(opts.A, a))
                 h_a, _ = trainer.gen_a.encode(images_a)
                 j = 0
                 for b in sample_b:

                     images_b = img_transform(os.path.join(opts.B, b))
                     _, n_b_fake = trainer.gen_b.encode(images_b)
                     for n in range(images_b.size(0)):
                         n_b = n_b_fake[n].unsqueeze(0)
                         outputs = trainer.gen_b.decode(h_a+n_b)
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

         else:
             i = 0
             for b in sample_b:
                 # print(a)
                 images_b = img_transform(os.path.join(opts.B, b))
                 h_b, _ = trainer.gen_b.encode(images_b)
                 j = 0
                 for a in sample_a:

                     images_a = img_transform(os.path.join(opts.A, a))
                     _, n_a_fake = trainer.gen_a.encode(images_a)
                     for n in range(images_a.size(0)):
                         n_a = n_a_fake[n].unsqueeze(0)
                         outputs = trainer.gen_a.decode(h_b+n_a)
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
     else:
         pass