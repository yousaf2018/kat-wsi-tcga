#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:yushan zheng
# emai:yszheng@buaa.edu.cn

import argparse
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

from model import KAT, kat_inference
from loader import KernelWSILoader
from loader import DistributedWeightedSampler
from utils import *

import random
import builtins
import warnings

def arg_parse():
    parser = argparse.ArgumentParser(description='KAT inference arguments.')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved KAT model checkpoint.')
    parser.add_argument('--cfg', type=str,
                        default='',
                        help='The path of yaml config file')

    parser.add_argument('--fold', type=int, default=-1, help='use all data for training if it is set -1')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers to load data.')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    
    return parser.parse_args()

def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    args.num_classes = args.task_list[args.label_id]['num_classes']
    graph_model_path = get_kat_path(args, args.prefix_name)

    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    args.start_epoch = checkpoint['epoch']
    args.input_dim = checkpoint['input_dim']
    
    model = KAT(
        num_pk=args.npk,
        patch_dim=args.input_dim,
        num_classes=args.num_classes, 
        dim=args.trfm_dim, 
        depth=args.trfm_depth, 
        heads=args.trfm_heads, 
        mlp_dim=args.trfm_mlp_dim, 
        dim_head=args.trfm_dim_head, 
        num_kernal=args.kn,
        pool=args.trfm_pool
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    train_set = KernelWSILoader(
        os.path.join(graph_model_path, 'train'),
        max_node_number=args.max_nodes,
        patch_per_kernel=args.npk,
        task_id=args.label_id,
        max_kernel_num=args.kn,
        node_aug=args.node_aug,
        two_augments=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, sampler=None
    )

    criterion = nn.CrossEntropyLoss().cuda(args.gpu) if args.gpu is not None else nn.CrossEntropyLoss()

    train_acc, train_cm, train_auc, train_data = evaluate(train_loader, model, criterion, args, 'Train')

    with open(os.path.join(graph_model_path, 'train_eval.pkl'), 'wb') as f:
        pickle.dump({'acc': train_acc, 'cm': train_cm, 'auc': train_auc, 'data': train_data}, f)

    print('Training dataset evaluation done.')

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
        
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f} Sample per Second {time:.3f}'
              .format(top1=top1, top2=top2, time=len(val_loader)*args.batch_size/processing_time))

    y_preds = torch.cat(y_preds)
    y_labels = torch.cat(y_labels)
    confuse_mat, auc = calc_classification_metrics(y_preds, y_labels, args.num_classes, prefix=prefix)
    print("Confusion matrix -->", confuse_mat)
    return top1.avg, confuse_mat, auc


if __name__ == "__main__":
    args = arg_parse()
    main(args)
