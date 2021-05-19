# -*- coding: utf-8 -*-
from __future__ import division
import utils.dataset_plain as data_processor
from utils import utils
import json
import pdb

import argparse
import logging
import torch
import torch.utils
import torch.backends.cudnn as cudnn
import os, sys
import numpy as np
import random

def parallel_test(model, data_loader, logger, show_interval=50):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(data_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            outputs = model(input)
        n = input.size(0)
        prec1, prec5 = utils.accuracy(outputs, target, topk=(1, 5))

        objs.update(0, n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        #if step % show_interval == 0:
            #logger.info('Valid %03d   Loss %.4f   Top@1 %.4f   Top@5 %.4f', step, objs.avg, top1.avg, top5.avg)

    logger.info('*All Valid %03d   Loss %.4f   Top@1 %.4f   Top@5 %.4f', step, objs.avg, top1.avg, top5.avg)

    return {'top1': top1.avg, 'top5': top5.avg, 'loss': objs.avg} 

def parallel_update_BN(model, data_loader, logger, show_interval=50):
    model.train()
    data_len = len(data_loader)
    for step, (input, target) in enumerate(data_loader):
        #target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            outputs = model(input)
        #if step % show_interval == 0:
            #logger.info('Update BN %d/%d..........', step, data_len)

def parse_args():
    parser = argparse.ArgumentParser("Model testing.")
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data', type=str, default='cifar100', help='dataset for the task')
    parser.add_argument('--model_code_path', type=str, help='the path of models')
    parser.add_argument('--update_bn_portion', type=float, default=0.1, help='as said')
    parser.add_argument('--work_dir', help='the dir to save logs')
    parser.add_argument('--to_net', action='store_true', help='trans_to_net')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--gpu', type=str, default='0,1', help='gpu device id')
    parser.add_argument('--silence', action='store_true', help='trans_to_net')
    parser.add_argument('--model_type', type=str, default='spn_plus', help='SPN type mode')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    #pdb.set_trace()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = list(map(int, args.gpu.split(',')))
    gpus_len = len(gpus)
    # create work_dir
    utils.create_exp_dir(args.work_dir)

    # init logger before other steps
    logger = utils.get_common_logger(log_path=os.path.join(args.work_dir, 'log.txt'), silence=args.silence)
    logger.info('DataParallel testing: {}')

    batch_size = args.batch_size * gpus_len
    num_workers =  16
    train_data, val_data, test_data = data_processor.get_dataset(args.data)
    data_loader = torch.utils.data.DataLoader(
      val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(args.update_bn_portion * num_train))
    #print(split)
    update_bn_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    data_loader_update = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=update_bn_sampler, pin_memory=True, num_workers=num_workers)
    logger.info('Test dataset is builded.')
    logger.info('-------------------------------------')

    if args.model_type == 'spn_prl_4812p1':
        from model.resnet_spn_prl_4812p1 import ResNet20_SPN
    elif args.model_type == 'spn_branch_multi_uprl_48p1':
        from model.resnet_spn_branch_multi_uniform_prl_48p1 import ResNet20_SPN
    elif args.model_type == 'spn_branch_multiall_prl_48p1':
        from model.resnet_spn_branch_multiall_prl_48p1 import ResNet20_SPN
    else:
        print('Please set model type')
        
    model =  ResNet20_SPN(100)
    model = torch.nn.DataParallel(model).cuda() 
    #logger.info('Classification model is builded.')
    model_pool = utils.load_json(args.model_code_path)
    state_dict = load_weights(model, args.checkpoint, logger)
    for key, value in model_pool.items():
        #logging.info('\n')
        #logging.info('\n')
        model_code_str = value['arch']
        logging.info("arch name: {}, code: {}".format(key, model_code_str))
        model_code = list(map(int, model_code_str.split('-')))

        model.load_state_dict(state_dict)

        if hasattr(model, 'module'):
            model.module.set_mask(model_code)
        else:
            model.set_mask(model_code)

        #logger.info('Update BN for candidate model root')
        parallel_update_BN(model, data_loader_update, logger)
        logger.info('Update BN is done')
        #logger.info('Model testing ...')
        outputs = parallel_test(model, data_loader, logger)
        model_pool[key]['acc'] = outputs['top1']

    json_base_name = os.path.basename(args.model_code_path)
    save_path = json_base_name[:-5] + '_evaluated.json'
    save_path = os.path.join(args.work_dir, save_path)
    with open(save_path, 'w') as f:
        json.dump(model_pool, f)

def load_weights(model, model_path, logging):
    if os.path.exists(model_path):
        logging.info('=> loading checkpoint %s' % model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        if list(state_dict.keys())[0].startswith('module.') and (not hasattr(model, 'module')):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        elif (not list(state_dict.keys())[0].startswith('module.')) and hasattr(model, 'module'):
            state_dict = {'module.'+k: v for k, v in checkpoint['state_dict'].items()}
        logging.info('=> loaded checkpoint %s' % model_path)
        return state_dict
    else:
        logging.info('model path %s is not exists.' % model_path)
        return None
    return None


if __name__ == '__main__':
    main()
