# -*- coding: utf-8 -*-
# !/usr/bin/env python3

'''
Divide face accordance MeGlass Attr type.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os
import pdb

output_path_train = "/home/bingwen/ERGAN-Pytorch/datasets/MeGlass"

image_path_A = "/home/bingwen/ERGAN-Pytorch/datasets/MeGlass/trainA"
image_path_B = "/home/bingwen/ERGAN-Pytorch/datasets/MeGlass/trainB"

gallery_glass = "/home/bingwen/ERGAN-Pytorch/datasets/MeGlass/gallery_black_glass.txt"
gallery_no_glass = "/home/bingwen/ERGAN-Pytorch/datasets/MeGlass/gallery_no_glass.txt"
probe_glass = "/home/bingwen/ERGAN-Pytorch/datasets/MeGlass/probe_black_glass.txt"
probe_no_glass = "/home/bingwen/ERGAN-Pytorch/datasets/MeGlass/probe_no_glass.txt"

def main():

    gallery_glass_dir = os.path.join(output_path_train, "gallery_glass")
    gallery_no_glass_dir = os.path.join(output_path_train, "gallery_no_glass")
    probe_glass_dir = os.path.join(output_path_train, "probe_glass")
    probe_no_glass_dir = os.path.join(output_path_train, "probe_no_glass")

    dirs = [gallery_glass_dir, gallery_no_glass_dir, probe_glass_dir, probe_no_glass_dir]

    for d in dirs:
        os.makedirs(d)

    count_A = 0
    count_B = 0
    count_C = 0
    count_D = 0

    with open(gallery_glass, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            img = info[0]
            img_path_old = os.path.join(image_path_B, img)
            shutil.copy(img_path_old, gallery_glass_dir)
            count_A += 1

    with open(gallery_no_glass, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            img = info[0]
            img_path_old = os.path.join(image_path_A, img)
            shutil.copy(img_path_old, gallery_no_glass_dir)
            count_B += 1

    with open(probe_glass, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            img = info[0]
            img_path_old = os.path.join(image_path_B, img)
            shutil.copy(img_path_old, probe_glass_dir)
            count_C += 1

    with open(probe_no_glass, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            img = info[0]
            img_path_old = os.path.join(image_path_A, img)
            shutil.copy(img_path_old, probe_no_glass_dir)
            count_D += 1

    print("gallery_glass have %d images!" % count_A)
    print("gallery_no_glass have %d images!" % count_B)
    print("probe_glass have %d images!" % count_C)
    print("probe_no_glass have %d images!" % count_D)

if __name__ == "__main__":
    main()