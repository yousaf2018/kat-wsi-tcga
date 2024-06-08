import argparse
import os
import torch
import numpy as np
from model import KAT, kat_inference
from loader import KernelWSILoader
from utils import *
from yacs.config import CfgNode

def arg_parse():
    parser = argparse.ArgumentParser(description='KAT inference.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model checkpoint')
    parser.add_argument('--cfg', type=str, required=True, help='Path to the yaml config file')
    parser.add_argument('--fold', type=int, required=True, help='Fold number to use for inference')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers to load data')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    return parser.parse_args()

def load_model(args):
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model_args = checkpoint['args']

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

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    model.eval()
    return model

def main():
    args = arg_parse()

    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    model = load_model(args)

    graph_list_dir = os.path.join(get_graph_list_path(args), 'list_fold_{}'.format(args.fold))
    test_path = os.path.join(graph_list_dir, 'test')

    dataset = KernelWSILoader(
        test_path,
        max_node_number=args.max_nodes,
        patch_per_kernel=args.npk,
        task_id=args.label_id,
        max_kernel_num=args.kn,
        node_aug=False,
        two_augments=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, sampler=None
    )

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    with torch.no_grad():
        top1 = AverageMeter('Acc@1', ':6.2f')
        top2 = AverageMeter('Acc@2', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        
        y_preds = []
        y_labels = []

        for i, (data, label) in enumerate(data_loader):
            target = label.cuda(non_blocking=True) if args.gpu is not None else label

            _, output = kat_inference(model, data)
            loss = criterion(output, target)
            y_preds.append(torch.nn.functional.softmax(output, dim=1).cpu().data)
            y_labels.append(label)

            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top2.update(acc2[0], target.size(0))

        y_preds = torch.cat(y_preds)
        y_labels = torch.cat(y_labels)
        confuse_mat, auc = calc_classification_metrics(y_preds, y_labels, args.num_classes, prefix='Test')
        
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
              .format(top1=top1, top2=top2))
        print("Confusion matrix -->", confuse_mat)
        print("AUC -->", auc)

if __name__ == "__main__":
    main()
