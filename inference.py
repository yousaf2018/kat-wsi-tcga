#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# email: yszheng@buaa.edu.cn
import time

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from yacs.config import CfgNode

from model import KAT, kat_inference
from loader import KernelWSILoader
from utils import *

import random
import warnings

def arg_parse():
    parser = argparse.ArgumentParser(description='KAT inference arguments.')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved KAT model checkpoint.')
    parser.add_argument('--cfg', type=str, required=True,
                        help='The path of yaml config file.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers to load data.')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed for initializing training.')
    parser.add_argument('--print-freq', type=int, default=1,
                        help='The mini-batch frequency to print results.')

    return parser.parse_args()

def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        print("Use GPU: {} for inference".format(args.gpu))
        torch.cuda.set_device(args.gpu)

    # Load the model checkpoint
    checkpoint = torch.load(args.model_path, map_location='cuda' if args.gpu is not None else 'cpu')
    model_args = checkpoint['args']

    # Initialize the model
    model = KAT(
        num_pk=model_args.npk,
        patch_dim=model_args.input_dim,
        num_classes=model_args.num_classes, 
        dim=model_args.trfm_dim, 
        depth=model_args.trfm_depth, 
        heads=model_args.trfm_heads, 
        mlp_dim=model_args.trfm_mlp_dim, 
        dim_head=model_args.trfm_dim_head, 
        num_kernal=model_args.kn,
        pool=model_args.trfm_pool
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    # Load the test data
    graph_list_dir = os.path.join(get_graph_list_path(model_args), model_args.fold_name)
    test_path = os.path.join(graph_list_dir, 'test')
    print("Test path -->", test_path)
    test_set = KernelWSILoader(test_path,
        max_node_number=model_args.max_nodes,
        patch_per_kernel=model_args.npk,
        task_id=model_args.label_id,
        max_kernel_num=model_args.kn,
        node_aug=False,
        two_augments=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, sampler=None
    )

    criterion = nn.CrossEntropyLoss().cuda(args.gpu) if args.gpu is not None else nn.CrossEntropyLoss()

    test_acc, test_cm, test_auc, test_data = evaluate(test_loader, model, criterion, model_args, 'Test')

    print(f"Test Accuracy: {test_acc}")
    print(f"Test Confusion Matrix: \n{test_cm}")
    print(f"Test AUC: {test_auc}")

    with open(os.path.join(os.path.dirname(args.model_path), 'test_eval.pkl'), 'wb') as f:
        pickle.dump({'acc': test_acc, 'cm': test_cm, 'auc': test_auc, 'data': test_data}, f)

    print('Inference on test dataset complete.')

def evaluate(val_loader, model, criterion, args, prefix='Test'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top2,
                             prefix=prefix)

    # switch to evaluate mode
    model.eval()
    y_preds = []
    y_labels = []
    end = time.time()
    
    processing_time = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            target = label.cuda(non_blocking=True) if args.gpu is not None else label
            # compute output
            pro_start = time.time()
            _, output = kat_inference(model, data)
            processing_time += (time.time() - pro_start)
            loss = criterion(output, target)

            y_preds.append(F.softmax(output, dim=1).cpu().data)
            y_labels.append(label)
            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top2.update(acc2[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f} Sample per Second {time:.3f}'
              .format(top1=top1, top2=top2, time=len(val_loader)*args.batch_size/processing_time))

    y_preds = torch.cat(y_preds)
    y_labels = torch.cat(y_labels)
    confuse_mat, auc = calc_classification_metrics(y_preds, y_labels, args.num_classes, prefix=prefix)
    print("Confusion matrix and auc-->", confuse_mat, auc)
    return top1.avg, confuse_mat, auc, {'pred':y_preds, 'label':y_labels}

if __name__ == "__main__":
    args = arg_parse()
    main(args)
