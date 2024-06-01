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
    parser = argparse.ArgumentParser(description='KAT arguments.')

    parser.add_argument('--cfg', type=str,
            default='',
            help='The path of yaml config file')

    parser.add_argument('--fold', type=int, default=-1, help='use all data for training if it is set -1')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers to load data.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--shuffle-train', default=False, action='store_true',
                        help='Shuffle the train list')
    parser.add_argument('--weighted-sample', action='store_true',
                        help='Balance the sample number from different types\
                              in each mini-batch for training.')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--redo', default=False, action='store_true',
                        help='Ingore all the cache files and re-train the model.')
    parser.add_argument('--eval-model', type=str, default='',
                        help='provide a path of a trained model to evaluate the performance')
    parser.add_argument('--eval-freq', type=int, default=30,
                        help='The epoch frequency to evaluate on vlidation and test sets.')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='The mini-batch frequency to print results.')
    parser.add_argument('--prefix-name', type=str, default='',
                        help='A prefix for the model name.')
    
    parser.add_argument('--node-aug', default=False, action='store_true',
                        help='Randomly reduce the nodes for data augmentation》')

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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.num_classes = args.task_list[args.label_id]['num_classes']
    graph_model_path = get_kat_path(args, args.prefix_name)

    checkpoint = []
    if not args.redo:
        checkpoint_path = os.path.join(
            graph_model_path, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device('cpu'))
            print("=> loading checkpoint")

    if checkpoint:
        args.start_epoch = checkpoint['epoch']
        if args.start_epoch >= args.num_epochs:
            print('model training is finished')
            return 0
        else:
            print('model train from epoch {}/{}'.format(args.start_epoch, args.num_epochs))
    else:
        args.start_epoch = 0

    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None and not args.distributed:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.rank == -1:
            if args.dist_url == "env://":
                args.rank = int(os.environ["RANK"])
            elif 'SLURM_PROCID' in os.environ:
                args.rank = int(os.environ['SLURM_PROCID'])
                
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    graph_list_dir = os.path.join(get_graph_list_path(args), args.fold_name)
    # train graph data
    train_set = KernelWSILoader(
            os.path.join(graph_list_dir, 'train'),
            max_node_number=args.max_nodes,
            patch_per_kernel=args.npk,
            task_id=args.label_id,
            max_kernel_num=args.kn,
            node_aug=args.node_aug,
            two_augments=False
            )

    args.input_dim = train_set.get_feat_dim()
    # create model
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
        pool = args.trfm_pool, 
    )

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    if os.path.isfile(args.resume):
        print("=> resume checkpoint '{}'".format(args.resume))
        resume_model_params = torch.load(
            args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(resume_model_params['state_dict'])
    else:
        if checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cudnn.benchmark = True

    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader, test_loader = None, None
    val_loader, test_loader = load_val_test_data(args)

    best_epoch = -1
    best_loss = float('inf')
    best_acc = 0

    log_dir = os.path.join(graph_model_path, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, '{}.txt'.format(args.label_id))
    logger = Logger(log_file)
    logger.write('{}\n'.format(str(args)))

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        is_best = False
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs - 1:
            val_loss, val_acc = validate(val_loader, model, criterion, args)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_loss < best_loss:
                is_best = True
                best_epoch = epoch
                best_loss = val_loss
                best_acc = val_acc

        logger.write('[{}] train_loss:{:.5f}, train_acc:{:.5f} val_loss:{:.5f}, val_acc:{:.5f}\n'.format(
            epoch, train_loss, train_acc, val_loss, val_acc))

        if args.gpu is None or (args.gpu == 0 and not args.multiprocessing_distributed):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc': best_acc,
            }, is_best, graph_model_path)

    if test_loader:
        model_best = model
        if args.gpu is not None:
            model_best = model_best.cuda(args.gpu)
        if args.gpu is None or (args.gpu == 0 and not args.multiprocessing_distributed):
            best_model_path = os.path.join(
                graph_model_path, 'model_best.pth.tar')
            if os.path.exists(best_model_path):
                print("=> loading best model")
                resume_model_params = torch.load(
                    best_model_path, map_location=torch.device('cpu'))
                model_best.load_state_dict(
                    resume_model_params['state_dict'])
            else:
                print("best model path not found")
            evaluate(test_loader, model_best, args)
    
    # Plot training and validation loss
    plt.figure()
    plt.plot(range(args.start_epoch, args.num_epochs), train_losses, label='Training Loss')
    plt.plot(range(args.start_epoch, args.num_epochs, args.eval_freq), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(range(args.start_epoch, args.num_epochs), train_accuracies, label='Training Accuracy')
    plt.plot(range(args.start_epoch, args.num_epochs, args.eval_freq), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    end = time.time()
    for i, (features, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            features = features.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)

        output = model(features)
        loss = criterion(output, label)

        acc1, = accuracy(output, label, topk=(1,))
        losses.update(loss.item(), features.size(0))
        top1.update(acc1[0], features.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    return losses.avg, top1.avg

def validate(val_loader, model, criterion, args):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        end = time.time()
        for i, (features, label) in enumerate(val_loader):
            if args.gpu is not None:
                features = features.cuda(args.gpu, non_blocking=True)
                label = label.cuda(args.gpu, non_blocking=True)

            output = model(features)
            loss = criterion(output, label)

            acc1, = accuracy(output, label, topk=(1,))
            losses.update(loss.item(), features.size(0))
            top1.update(acc1[0], features.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def evaluate(test_loader, model, args):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for i, (features, label) in enumerate(test_loader):
            if args.gpu is not None:
                features = features.cuda(args.gpu, non_blocking=True)

            output = model(features)
            preds.append(output.cpu().numpy())
            labels.append(label.numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    results = kat_inference(
        preds, labels, threshold=args.inference_threshold)
    print("Test Results: ", results)
    return results

if __name__ == "__main__":
    args = arg_parse()
    main(args)
