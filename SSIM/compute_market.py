import argparse
import os
from IPython import embed
from util import util
import models.dist_model as dm
import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str, default='../Market/pytorch/train_all')
opt = parser.parse_args()

## Initializing the model
model = dm.DistModel()
model.initialize(model='net-lin',net='vgg',use_gpu=True)
#model.initialize(model='ssim',use_gpu=True)

score = 0
num = 0

score_max = 0
score_min = 1

for ii in range(10):
	for subdir in os.listdir(opt.dir):
		subdir = opt.dir + '/'+ subdir
		count = 0
		all_file = os.listdir(subdir)
		for i in range(20):
			randp = np.random.permutation(len(all_file))
			rand1 = randp[0]
			rand2 = randp[1]
			# Load images
			img0 = util.im2tensor(util.load_image(os.path.join(subdir,all_file[rand1]))) # RGB image from [-1,1]
			img1 = util.im2tensor(util.load_image(os.path.join(subdir,all_file[rand2])))
			# mask = torch.zeros(img0.shape)
			# mask[:, :, round((224/192)* 60):round((224/192) * 140), round((224/192) * 32):round((224/192)* 168)] = 1
			# img0 = img0 * mask
			# img1 = img1 * mask
			# Compute distance
			# dist01 = model.forward(img0[:, :, 92 :144, 48 :172],
			#    					   img1[:, :, 92 :144, 48 :172]) # celebA

			#dist01 = model.forward(img0[:, :, 68:104, 48:154],
			#					   img1[:, :, 68:104, 48:154])  # lfw

			dist01 = model.forward(img0[:, :, 48:78, 24:96],
								   img1[:, :, 48:78, 24:96])  # meglass
			#dist01 = model.forward(img0,img1)
			num +=1
			score +=dist01
	print('%d::%.6f'%(ii,score/num))
	if score/num >score_max:
		score_max = score/num
	if score/num <score_min:
		score_min = score/num
print('score: %.4f +- %.5f'%((score_max+score_min)/2, (score_max-score_min)/2) )
