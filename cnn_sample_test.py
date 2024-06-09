#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2022/06
# author:Yushan Zheng
# email:yszheng@buaa.edu.cn

import numpy as np
import pickle
import os
import cv2
import argparse
from multiprocessing import Pool
from yacs.config import CfgNode
from loader import get_tissue_mask, extract_tile
from utils import *

parser = argparse.ArgumentParser('Sampling patches for CNN training')
parser.add_argument('--cfg', type=str, default='',
                    help='The path of yaml config file')

parser.add_argument('--num-workers', type=int, default=8,
                    help='The processors used for parallel sampling.')
parser.add_argument('--ignore-annotation', action='store_true', default=False,
                    help='Ignore annotations when sampling.')
parser.add_argument('--invert-rgb', action='store_true', default=False,
                    help='Adjust the format between RGB and BGR.\
                        The default color format of the patch is BGR')

def main(args):
    np.random.seed(1)
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    with open(args.slide_list, 'rb') as f:
        slide_data = pickle.load(f)
    slide_list = slide_data['test']

    args.dataset_path = get_sampling_path(args)
    print('slide num', len(slide_list))

    sampling_list = [(i, args) for i in slide_list]
    if args.num_workers < 2:
        # sampling the data using single thread
        for s in slide_list:
            sampling_slide((s, args))
    else:
        # sampling the data in parallel
        with Pool(args.num_workers) as p:
            p.map(sampling_slide, sampling_list)

    return 0

def extract_and_save_tiles(image_dir, slide_save_dir, position_list, tile_size,
                           imsize, step, invert_rgb=False):
    for pos in position_list:
        img = extract_tile(image_dir, tile_size, pos[1] * step, pos[0] * step,
                           imsize, imsize)
        
        if len(img) > 0:
            if invert_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(slide_save_dir, '{:04d}_{:04d}.jpg'.format(pos[1], pos[0])), img)

def sampling_slide(slide_info):
    slide_guid, slide_rpath, slide_label = slide_info[0]
    args = slide_info[1]

    time_file_path = os.path.join(args.dataset_path, slide_guid, 'info.txt')
    # if os.path.exists(time_file_path):
    #     print(slide_guid, 'is already sampled. skip.')
    #     return 0

    slide_path = os.path.join(args.slide_dir, slide_rpath)
    image_dir = os.path.join(slide_path, scales[args.level])

    tissue_mask = get_tissue_mask(cv2.imread(
            os.path.join(slide_path, 'Overview.jpg')))
    
    content_mat = cv2.blur(tissue_mask, ksize=args.filter_size, anchor=(0, 0))
    content_mat = content_mat[::args.srstep, ::args.srstep]
    
    mask_path = os.path.join(slide_path, 'AnnotationMask.png')
    # Use the annotation to decide the label of the patch if annotation is available.
    # Otherwise, assign a psudo-label to the patch based on the WSI label it belongs to.
    if not args.ignore_annotation and os.path.exists(mask_path):
        mask = cv2.imread(os.path.join(slide_path, 'AnnotationMask.png'), 0)
        positive_mat = cv2.blur(
            (mask > 0)*255, ksize=args.filter_size, anchor=(0, 0))
        positive_mat = positive_mat[::args.srstep, ::args.srstep]

        # the left-top position of benign patches
        bn_lt = np.transpose(
            np.asarray(
                np.where((positive_mat < args.negative_ratio * 255)
                        & (content_mat > args.intensity_thred)), np.int32))
        if bn_lt.shape[0] > args.max_per_class:
            bn_lt = bn_lt[np.random.choice(
                bn_lt.shape[0], args.max_per_class, replace=False)]

        if bn_lt.shape[0] > 0:
            slide_save_dir = os.path.join(args.dataset_path, slide_guid, '0')
            if not os.path.exists(slide_save_dir):
                os.makedirs(slide_save_dir)

            extract_and_save_tiles(image_dir, slide_save_dir, bn_lt,
                                args.tile_size, args.imsize, args.sample_step, args.invert_rgb)

        class_list = np.unique(mask[mask > 0])
        for c in class_list:
            class_index_mat = cv2.blur(
                (mask == c)*255, ksize=args.filter_size, anchor=(0, 0))
            class_index_mat = class_index_mat[::args.srstep, ::args.srstep]

            # the left-top position of tumor patches
            tm_lt = np.transpose(
                np.asarray(
                    np.where((class_index_mat > args.positive_ratio * 255)
                            & (content_mat > args.intensity_thred)), np.int32))

            if tm_lt.shape[0] > args.max_per_class:
                tm_lt = tm_lt[np.random.choice(
                    tm_lt.shape[0], args.max_per_class, replace=False)]

            slide_save_dir = os.path.join(args.dataset_path, slide_guid, str(c))
            if not os.path.exists(slide_save_dir):
                os.makedirs(slide_save_dir)

            extract_and_save_tiles(image_dir, slide_save_dir, tm_lt,
                                args.tile_size, args.imsize, args.sample_step, args.invert_rgb)

            if args.save_mask:
                extract_and_save_tiles(mask, slide_save_dir, tm_lt, args)
    else:
        content_lt = np.transpose(
            np.asarray(
                np.where(content_mat > args.intensity_thred), np.int32))
        if content_lt.shape[0] > args.max_per_class:
            content_lt = content_lt[np.random.choice(
                content_lt.shape[0], args.max_per_class, replace=False)]
        
        if content_lt.shape[0] > 0:
            slide_save_dir = os.path.join(args.dataset_path, slide_guid, str(slide_label))
            if not os.path.exists(slide_save_dir):
                os.makedirs(slide_save_dir)

            extract_and_save_tiles(image_dir, slide_save_dir, content_lt,
                                args.tile_size, args.imsize, args.sample_step, args.invert_rgb)

        print(slide_guid, 'Patch num: ', content_lt.shape[0])

    if os.path.exists(os.path.join(args.dataset_path, slide_guid)):
        
        with open(time_file_path, 'w') as f:
            f.write('Sampling finished')
   


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)