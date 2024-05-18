#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2022/06
# author: Yushan Zheng
# email: yszheng@buaa.edu.cn

import numpy as np
import pickle
import os
import cv2
import argparse
from multiprocessing import Pool
from yacs.config import CfgNode
from openslide import OpenSlide

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
    slide_list = slide_data['train'] + slide_data['test']

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

    list_path = get_data_list_path(args)
    dataset_split_path = os.path.join(list_path, 'split.pkl')
    if not os.path.exists(dataset_split_path):
        train_list = slide_data['train'] 
        np.random.shuffle(train_list)

        folds = [train_list[f_id::args.fold_num] for f_id in range(args.fold_num)]
        folds.append(slide_data['test'] )

        if not os.path.exists(list_path):
            os.makedirs(list_path)

        with open(dataset_split_path, 'wb') as f:
            pickle.dump(folds, f)

    make_list(args)

    return 0

def sampling_slide(slide_info):
    slide_guid, slide_rpath, slide_label = slide_info[0]
    print("Here is slide to process -->", slide_guid, slide_rpath, slide_label)
    args = slide_info[1]

    time_file_path = os.path.join(args.dataset_path, slide_guid, 'info.txt')
    if os.path.exists(time_file_path):
        print(slide_guid, 'is already sampled. skip.')
        return 0

    for root, dirs, files in os.walk(args.slide_dir):
        for file in files:
            if file.endswith(".svs") and file.split(".svs")[0] == slide_rpath:
                slide_path = os.path.join(root, file.split('.svs')[0])
                break
            
    # Here, you can directly process the SVS images without creating an overview image
    # For example, you can extract and save tiles from the SVS image directly

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
