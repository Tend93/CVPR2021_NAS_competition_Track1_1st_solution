# -*- coding: utf-8 -*-
# Tend used
import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.backends.cudnn as cudnn
import shutil
import random
import warnings
import utils.dataset_plain as data_processor
from utils import utils
import json, pdb

def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', 1]:
        return True
    elif v.lower() in ['false', 'no', 'f', 'n', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser("backbone")
parser.add_argument('--fix_pretrain', type=str2bool, default=False, help='just test the model')
parser.add_argument('--dif_lr', type=str2bool, default=False, help='just test the model')
parser.add_argument('--data', metavar='DIR', help='dataset for the task')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--sample_num', type=int, default=1, help='sample random path')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--gamma', type=float, default=0.1, help='decay ratio for scheduler')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='2', help='gpu device id')
parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
parser.add_argument('--step_epochs', type=int, default=50, help='number of epochs for step lr descend.')
parser.add_argument('--step_list_ratio', type=str, default='', help='the list of epochs for step lr descend.')
parser.add_argument('--lr_adjust_type', type=str, default='cosine', help='type of lr adjust')
parser.add_argument('--model_type', type=str, default='spn_branch', help='SPN type mode')
parser.add_argument('--distill_coeff', type=float, default=1.0, help='')

parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='DEFAULT', help='experiment name')
parser.add_argument('--save_iter', type=int, default=10, help='')
parser.add_argument('--root_path', type=str, default='', help='path to save the model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')

######## finetune a model #####################################
parser.add_argument('--source_model', type=str, help='the path of trained model for finetune')

def setup_logger(log_dir):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    # don't log results for the non-master process
    log_file = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

### setting some things #####################################

IMAGE_SIZE = 224
NUMCLASS_LIB = {'imagenet': 1000}

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    ### create file path #####################
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join(args.root_path, args.save)
    utils.create_exp_dir(args.save)

    if len(args.step_list_ratio) > 0 and args.lr_adjust_type == 'step':
        args.step_list_ratio = list(map(float, args.step_list_ratio.split('-')))
        args.step_list = [int(float(args.epochs)*i) for i in args.step_list_ratio]

    print('gpu device = %s' % args.gpu)
    print("args = %s", args)

    main_worker(args)

def main_worker(args):
    ### setting seed and cudnn ##################################
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = setup_logger(args.save)
    logging.info('gpu device = %s' % args.gpu)
    logging.info(args)

    fjson = {}
    fjson['top1']       = 0.0
    fjson['top5']       = 0.0
    fjson['flops']      = 0.0
    fjson['params']     = 0.0
    fjson['infer_time'] = 0.0
    fjson['bw']         = 0.0

    fjson['args'] = args.__dict__


    # from model.resnet_spn_branch_multi import ResNet20_SPN
    # from model.resnet_spn_branch_multi2 import ResNet20_SPN as ResNet20_SPN2
    from model.resnet_spn_branch_multi_uniform_prl_48p1 import ResNet20_SPN
    from model.resnet_spn_branch_multiall_prl_48p1 import ResNet20_SPN as ResNet20_SPN2

    ###### create model ###################
    pre_model = ResNet20_SPN(100)
    model = ResNet20_SPN2(100)
    #criterion = CrossEntropyKLLoss()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    logging.info(model)

    model = model.cuda()
    pre_model = pre_model.cuda()
    if args.dif_lr:
        params_lr = [{'params': model.conv1.parameters(), 'lr': args.learning_rate},
                     {'params': model.layer1.parameters(), 'lr': args.learning_rate},
                     {'params': model.layer2.parameters(), 'lr': args.learning_rate * 2},
                     {'params': model.layer3.parameters(), 'lr': args.learning_rate * 4},
                     {'params': model.fc_twin.parameters(), 'lr': args.learning_rate * 4},
                     {'params': model.fc.parameters(), 'lr': args.learning_rate}]
    else:
        params_lr = model.parameters()
    optimizer = torch.optim.SGD(params_lr,
                              args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=False)

    if args.lr_adjust_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.learning_rate_min)
    else:
        if len(args.step_list_ratio) > 0:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step_list, gamma=args.gamma, last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_epochs, gamma=args.gamma, last_epoch=-1)

    if args.source_model is not None:
        checkpoint = utils.load_checkpoint(pre_model, args.source_model, logging)
        model.load_weight_from_previous_stage(pre_model)
        checkpoint = None
        torch.cuda.empty_cache()

    train_data, val_data, test_data = data_processor.get_dataset(args.data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    val_queue = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    for epoch in range(args.start_epoch, args.epochs):

        logging.info('epoch %d ########################################', epoch)
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        # training

        train_top1, train_top5, train_obj = train(train_queue, model, optimizer, args, logging, criterion)
        logging.info(' * Train top@1 %f top@5 %f', train_top1, train_top5)

        scheduler.step()  # update lr

        top1_acc = train_top1
        top5_acc = train_top5

        if (epoch + 1) % args.save_iter == 0:
            states = {'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'state_dict': model.state_dict(),
                      'epoch': epoch,
                      'top1': float(top1_acc),
                      'top5': float(top5_acc)}
            cur_weight_name = 'checkpoint_ep{}.pt'.format(epoch)
            torch.save(states, os.path.join(args.save, cur_weight_name))

def train(train_queue, model, optimizer, args, logging, criterion):
    objs_cls = utils.AvgrageMeter()
    objs_dt = utils.AvgrageMeter()
    st_top1 = utils.AvgrageMeter()
    st_top5 = utils.AvgrageMeter()
    th_top1 = utils.AvgrageMeter()
    th_top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):

        n = input.size(0)
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        optimizer.zero_grad()
        model.set_use_branch(False)
        model.set_max_path()
        th_outputs = model(input)
        loss = criterion(th_outputs, target)
        th_prec1, th_prec5 = utils.accuracy(th_outputs, target, topk=(1, 5))
        objs_cls.update(loss.item(), n)
        th_top1.update(th_prec1.item(), n)
        th_top5.update(th_prec5.item(), n)
        if not args.fix_pretrain:
            loss.backward()

        with torch.no_grad():
            soft_label = F.softmax(th_outputs.detach(), dim=1)
        
        #distill_coeff = 1.0
        model.set_use_branch(True)
        for sample_i in range(args.sample_num):  
            if hasattr(model, 'module'):
                path_code = model.module.get_random_path()
                model.module.set_mask(path_code)                
            else:
                path_code = model.get_random_path()
                model.set_mask(path_code)
            st_outputs = model(input)

            loss = criterion(st_outputs, target)
            loss = (1 - args.distill_coeff) * loss + args.distill_coeff * torch.mean(torch.sum(- soft_label * nn.LogSoftmax(dim=1)(st_outputs), 1))
            loss.backward()

            st_prec1, st_prec5 = utils.accuracy(st_outputs, target, topk=(1, 5))
            objs_dt.update(loss.item(), n)
            st_top1.update(st_prec1.item(), n)
            st_top5.update(st_prec5.item(), n)

        optimizer.step()          #SGD update parameters

        if step % args.report_freq == 0:
            logging.info('Train %03d cl-Loss %.4f dt-Loss %.4f ST-Top@1 %.4f ST-Top@5 %.4f TH-Top@1 %.4f TH-Top@5 %.4f', \
                         step, objs_cls.avg, objs_dt.avg, st_top1.avg, st_top5.avg, th_top1.avg, th_top5.avg)

    return st_top1.avg, st_top5.avg, objs_cls.avg


# def infer(test_queue, model, args, logging, criterion):
#     objs = utils.AvgrageMeter()
#     top1 = utils.AvgrageMeter()
#     top5 = utils.AvgrageMeter()
#     model.eval()

#     for step, (input, target) in enumerate(test_queue):
#         target = target.cuda(non_blocking=True)
#         input = input.cuda(non_blocking=True)

#         if hasattr(model, 'module'):
#             model.module.set_random_path()
#         else:
#             model.set_random_path()

#         with torch.no_grad():
#             outputs = model(input)

#         n = input.size(0)
#         prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
#         objs.update(0, n)
#         top1.update(prec1.item(), n)
#         top5.update(prec5.item(), n)

#         if step % args.report_freq == 0:
#             logging.info('Valid %03d   Loss %f   Top@1 %f   Top@5 %f', step, objs.avg, top1.avg, top5.avg)

#     return top1.avg, top5.avg, objs.avg

if __name__ == '__main__':
    main() 

