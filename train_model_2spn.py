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
import torch.utils
import torch.distributed as dist
import torch.multiprocessing as mp
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
parser.add_argument('--test_stage', type=str2bool, default=False, help='just test the model')
parser.add_argument('--data', metavar='DIR', help='dataset for the task')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='2', help='gpu device id')
parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
parser.add_argument('--step_epochs', type=int, default=50, help='number of epochs for step lr descend.')
parser.add_argument('--step_list_ratio', type=str, default='', help='the list of epochs for step lr descend.')
parser.add_argument('--lr_adjust_type', type=str, default='cosine', help='type of lr adjust')
parser.add_argument('--model_type', type=str, default='spn_plus', help='SPN type mode')

parser.add_argument('--model_code', type=list, help='the code list of model strcture')
parser.add_argument('--model_code_str', type=str, default=None, help='the string code of model strcture')
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

######## start from a trained model ###########################
parser.add_argument('--start_epoch', type=int, default=0, help='the epoch when start')
parser.add_argument('--resume', type=str2bool, default=False, help='whether start from a trained model')
parser.add_argument('--trained_model', type=str, help='the path of trained model for continuing the train')

####### for multi-GPUs Distributed #############################
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str,
                    help='distributed backend')

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
    if not args.resume:
        args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join(args.root_path, args.save)
        utils.create_exp_dir(args.save)
    else:
        args.save = args.trained_model

    if len(args.step_list_ratio) > 0 and args.lr_adjust_type == 'step':
        args.step_list_ratio = list(map(float, args.step_list_ratio.split('-')))
        args.step_list = [int(float(args.epochs)*i) for i in args.step_list_ratio]

    if args.model_code_str is not None:
        args.model_code = list(map(int, args.model_code_str.split('-')))

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
    fjson['name']       = args.model_code_str
    fjson['top1']       = 0.0
    fjson['top5']       = 0.0
    fjson['flops']      = 0.0
    fjson['params']     = 0.0
    fjson['infer_time'] = 0.0
    fjson['bw']         = 0.0

    fjson['args'] = args.__dict__

    from model.resnet2spn import ResNet20

    ###### create model ###################
    model = ResNet20(100, args.model_code)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #logging.info(model)

    model = model.cuda()
    logging.info(model)
    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)

    if args.lr_adjust_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.learning_rate_min)
    else: 
        if len(args.step_list_ratio) > 0:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step_list, gamma=0.1, last_epoch=-1)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_epochs, gamma=0.1, last_epoch=-1)

    if args.resume:
        logging.info('resume the training from')
        logging.info(args.trained_model)
        checkpoint = load_checkpoint(os.path.join(args.trained_model + 'checkpoint_lastest.pt'), logging)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            fjson['top1'] = checkpoint['top1']
            fjson['top5'] = checkpoint['top5']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
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
        # validation
        val_top1, val_top5, val_obj = infer(val_queue, model, args, logging, criterion)
        logging.info(' ** Valid top@1 %f top@5 %f', val_top1, val_top5)
        top1_acc = val_top1
        top5_acc = val_top5

        states = {'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'state_dict': model.state_dict(),
                  'epoch': epoch,
                  'top1': float(top1_acc),
                  'top5': float(top5_acc)}
        torch.save(states, os.path.join(args.save, 'checkpoint_lastest.pt'))

        if top1_acc > fjson['top1']:
            shutil.copyfile(os.path.join(args.save, 'checkpoint_lastest.pt'), os.path.join(args.save, 'checkpoint_best.pt'))

        if top1_acc > fjson['top1']:
            fjson['top1'] = float(top1_acc)
            fjson['top5'] = float(top5_acc)

        with open(os.path.join(args.save, 'result_info.json'), 'w') as f:
            json.dump(fjson, f)

def load_checkpoint(model_path, logging):
    if os.path.exists(model_path):
        logging.info('=> loading checkpoint %s' % model_path)
        state = torch.load(model_path, map_location='cpu')
        logging.info('=> loaded checkpoint %s' % model_path)
        return state
    else:
        logging.info('model path %s is not exists.' % model_path)
        return None


def train(train_queue, model, optimizer, args, logging, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()          #SGD update parameters 
        prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train %03d   Loss %f   Top@1 %f   Top@5 %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def infer(test_queue, model, args, logging, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        with torch.no_grad():
            outputs = model(input)

        n = input.size(0)
        loss = criterion(outputs, target)
        prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid %03d   Loss %f   Top@1 %f   Top@5 %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg

if __name__ == '__main__':
    main()