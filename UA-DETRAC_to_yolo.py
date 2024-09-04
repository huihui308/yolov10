################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2019-2021 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# rm -rf ./test_output_class5;python3 UA-DETRAC_to_yolo.py --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/detect/UA-DETRAC --output_dir=./test_output_class5

# rm -rf ./test_output_class11;python3 UA-DETRAC_to_yolo.py --class_num=11 --target_width=1920 --target_height=1080 --input_dir=/home/david/dataset/class11-UA-DETRAC/UA-DETRAC --output_dir=/home/david/dataset/class11-UA-DETRAC
#
################################################################################

""" Script to prepare resized images/labels for primary detect. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import json
from tqdm import tqdm
#from PIL import Image
from typing import List
from xml.dom.minidom import parse
import os, sys, math, shutil, random, datetime, signal, argparse


categories4_list = ['person', 'rider', 'tricycle', 'car']
categories5_list = ['person', 'rider', 'tricycle', 'car', 'lg']
categories7_list = ['person', 'rider', 'tricycle', 'car', 'R', 'G', 'Y']
categories11_list = ['person', 'bicycle', 'motor', 'tricycle', 'car', 'bus', 'truck', 'plate', 'R', 'G', 'Y']
TQDM_BAR_FORMAT = '{l_bar}{bar:40}| {n_fmt}/{total_fmt} {elapsed}'


def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


def term_sig_handler(signum, frame)->None:
    sys.stdout.write('\r>> {}: \n\n\n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.write('\r>> {}: Catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    sys.stdout.write('\r>> {}: \n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.flush()
    os._exit(0)
    return


def parse_input_args(args = None):
    """ parse the arguments. """
    parser = argparse.ArgumentParser(description = 'Prepare resized images/labels dataset for LPD')
    parser.add_argument(
        "--input_dir",
        type = str,
        required = True,
        help = "Input directory to OpenALPR's benchmark end2end us license plates."
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        required = True,
        help = "Ouput directory to resized images/labels."
    )
    parser.add_argument(
        "--class_num",
        type = int,
        required = True,
        help = "Class num. 4:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3'}, 5:{'person':'0', 'rider':'1', 'tricycle':'2', 'car':'3', 'lg':'4'}, 6:{'person':'0', 'rider':'1', 'car':'2', 'R':'3', 'G':'4', 'Y':'5'}, 11:{'person':'0', 'bicycle':'1', 'motorbike':'2', 'tricycle':'3', 'car':'4', 'bus':'5', 'truck':'6', 'plate':'7', 'R':'8', 'G':'9', 'Y':'10'}"
    )
    parser.add_argument(
        "--target_width",
        type = int,
        required = True,
        help = "Target width for resized images/labels."
    )
    parser.add_argument(
        "--target_height",
        type = int,
        required = True,
        help = "Target height for resized images/labels."
    )
    return parser.parse_args(args)


def make_ouput_dir(output_dir:str)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lop_dir0 in ("train", "val"):
        first_dir = os.path.join(output_dir, lop_dir0)
        if not os.path.exists(first_dir):
            #shutil.rmtree(first_dir)
            os.makedirs(first_dir)
        for lop_dir1 in ("images", "labels"):
            second_dir = os.path.join(first_dir, lop_dir1)
            if not os.path.exists(second_dir):
                os.makedirs(second_dir)
    return


def uadetrac2_class_yolo_data(fp, one_target, obj_cnt_list, img_width, img_height)->None:
    type_str = None
    box_attr = one_target.getElementsByTagName('box')[0]
    obj_left = float( box_attr.getAttribute('left') )
    obj_top = float( box_attr.getAttribute('top') )
    obj_width = float( box_attr.getAttribute('width') )
    obj_height = float( box_attr.getAttribute('height') )
    type_attr = one_target.getElementsByTagName('attribute')[0]
    vehicle_type = type_attr.getAttribute('vehicle_type')
    #print(obj_left, obj_top, obj_width, obj_height, vehicle_type)
    if (len(obj_cnt_list) == 4) or (len(obj_cnt_list) == 5) or (len(obj_cnt_list) == 7):
        if vehicle_type in ('car', 'van', 'bus'):
            type_str = '3'
            obj_cnt_list[3] += 1
        else:
            #prRed('vehicle_type \'{}\' err'.format(vehicle_type))
            return
    elif len(obj_cnt_list) == 11:
        if vehicle_type in ('car', 'van'):
            type_str = '4'
            obj_cnt_list[4] += 1
        elif vehicle_type in ('bus'):
            type_str = '5'
            obj_cnt_list[5] += 1
        else:
            #prRed('vehicle_type \'{}\' err'.format(vehicle_type))
            return
    else:
        prRed('len(obj_cnt_list): {}'.format(len(obj_cnt_list)))
    x_center = (obj_left + obj_width/2)/img_width
    y_center = (obj_top + obj_height/2)/img_height
    yolo_width = obj_width/img_width
    yolo_height = obj_height/img_height
    if (x_center <= 0.0) or (y_center <= 0.0) or (yolo_width <= 0.0) or (yolo_height <= 0.0):
        prRed('Yolo pos {} {} {} {} err, return'.format(x_center, y_center, yolo_width, yolo_height))
        return
    fp.write("{} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(type_str, x_center, y_center, yolo_width, yolo_height))
    return


def deal_one_image_label_files(
        class_num, 
        train_fp, 
        val_fp, 
        img_file:str, 
        #label_file:str, 
        targets, 
        output_dir:str, 
        deal_cnt:int, 
        output_size:List[int], 
        obj_cnt_list
)->None:
    save_image_dir = None
    save_label_dir = None
    save_fp = None
    if ((deal_cnt % 10) >= 8):
        save_fp = val_fp
        save_image_dir = os.path.join(output_dir, "val/images")
        save_label_dir = os.path.join(output_dir, "val/labels")
    else:
        save_fp = train_fp
        save_image_dir = os.path.join(output_dir, "train/images")
        save_label_dir = os.path.join(output_dir, "train/labels")
    dir_name, full_file_name = os.path.split(img_file)
    sub_dir_name0, sub_dir_name1 = dir_name.split('/')[-2], dir_name.split('/')[-1]
    #print(sub_dir_name0, sub_dir_name1)
    save_file_name = sub_dir_name1 + "_" + os.path.splitext(full_file_name)[0] + "_" + str(random.randint(0, 999999999999)).zfill(12)
    #print(save_file_name)
    img = cv2.imread(img_file)
    (img_height, img_width, _) = img.shape
    resave_file = os.path.join(save_image_dir, save_file_name + ".jpg")
    os.symlink(img_file, resave_file)
    #shutil.copyfile(img_file, resave_file)
    save_fp.write(resave_file + '\n')
    save_fp.flush()
    #------
    with open(os.path.join(save_label_dir, save_file_name + ".txt"), "w") as fp:
        for one_target in targets:
            uadetrac2_class_yolo_data(fp, one_target, obj_cnt_list, img_width, img_height)
    return


def deal_dir_files(
        class_num, 
        deal_dir:str, 
        output_dir:str, 
        output_size:List[int], 
        obj_cnt_list
)->None:
    train_fp = open(output_dir + "/train.txt", "a+")
    val_fp = open(output_dir + "/val.txt", "a+")
    #------
    xml_dir = os.path.join(deal_dir, 'xml')
    images_dir = os.path.join(deal_dir, 'images')
    #print(deal_dir, xml_dir, images_dir)
    # get image and xml tuple list
    for root, dirs, files in os.walk(xml_dir):
        #print(len(files))
        pbar = enumerate(files)
        pbar = tqdm(pbar, total=len(files), desc="Processing {0:>15}".format(deal_dir.split('/')[-1]), colour='blue', bar_format=TQDM_BAR_FORMAT)
        for (i, one_file) in pbar:
        #for one_file in files:
            #print(os.path.splitext(one_file)[-1])
            file_name, file_type = os.path.splitext(one_file)
            #print(file_name, file_type)
            if file_type != '.xml':
                prRed('File {} formate error'.format(one_file))
                continue
            xml_image_dir = os.path.join(images_dir, file_name)
            if not os.path.exists(xml_image_dir):
                prRed('Image dir {} not exist'.format(xml_image_dir))
            #
            xml_dir_file = os.path.join(root, one_file)
            dom = parse(xml_dir_file)
            data = dom.documentElement
            frames = data.getElementsByTagName('frame')
            for one_frame in frames:
                image_file_name = 'img' + str( one_frame.getAttribute('num') ).zfill(5) + '.jpg'
                img_file = os.path.join(xml_image_dir, image_file_name)
                if not os.path.exists(img_file):
                    prRed('Image dir {} not exist'.format(img_file))
                #print(img_file)
                targets = one_frame.getElementsByTagName('target')
                deal_one_image_label_files(class_num, train_fp, val_fp, img_file, targets, output_dir, i, output_size, obj_cnt_list)
    train_fp.close()
    val_fp.close()
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_input_args(args)
    output_size = (args.target_width, args.target_height)
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))
    make_ouput_dir(args.output_dir)
    #------
    categories_list = None
    if args.class_num == 4:
        categories_list = categories4_list
    elif args.class_num == 5:
        categories_list = categories5_list
    elif args.class_num == 7:
        categories_list = categories7_list
    elif args.class_num == 11:
        categories_list = categories11_list
    else:
        prRed('Class num {} err, return'.format(args.class_num))
        return
    obj_cnt_list = [0 for _ in range( len(categories_list) )]
    deal_dir_list = ['train', 'test']
    for lop_dir in deal_dir_list:
        deal_dir_files(args.class_num, os.path.join(args.input_dir, lop_dir), args.output_dir, output_size, obj_cnt_list)
    print("\n")
    for category in categories_list:
        print("%10s " %(category), end='')
    print("%10s" %('sum'))
    for i in range( len(categories_list) ):
        print("%10d " %(obj_cnt_list[i]), end='')
    print("%10d" %(sum(obj_cnt_list)))
    print("\n")
    sys.stdout.write('\r>> {}: Generate yolov dataset success, save dir:{}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.output_dir))
    sys.stdout.flush()
    return


if __name__ == "__main__":
    main_func()