# -*- coding: utf-8 -*-
# !/usr/bin/env python3

'''
Divide face accordance CelebA Attr type.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os

output_path = "/home/bingwen/Dataset/CelebA/"
image_path = "/home/bingwen/Dataset/CelebA/img_align_celeba"
CelebA_Attr_file = "/home/bingwen/Dataset/CelebA/list_attr_celeba.txt"
Attr_type = 16  # Eyeglasses


def main():
    '''Divide face accordance CelebA Attr eyeglasses label.'''
    trainA_dir = os.path.join(output_path, "trainA")
    trainB_dir = os.path.join(output_path, "trainB")
    if not os.path.isdir(trainA_dir):
        os.makedirs(trainA_dir)
    if not os.path.isdir(trainB_dir):
        os.makedirs(trainB_dir)

    not_found_txt = open(os.path.join(output_path, "not_found_img.txt"), "w")

    count_A = 0
    count_B = 0
    count_N = 0

    with open(CelebA_Attr_file, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:]
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(image_path, filename)
            if os.path.isfile(filepath_old):
                if int(info[Attr_type]) == 1:
                    filepath_new = os.path.join(trainA_dir, filename)
                    shutil.copyfile(filepath_old, filepath_new)
                    count_A += 1
                else:
                    filepath_new = os.path.join(trainB_dir, filename)
                    shutil.copyfile(filepath_old, filepath_new)
                    count_B += 1
                print("%d: success for copy %s -> %s" % (index, info[Attr_type], filepath_new))
            else:
                print("%d: not found %s\n" % (index, filepath_old))
                not_found_txt.write(line)
                count_N += 1

    not_found_txt.close()

    print("TrainA have %d images!" % count_A)
    print("TrainB have %d images!" % count_B)
    print("Not found %d images!" % count_N)


if __name__ == "__main__":
    main()