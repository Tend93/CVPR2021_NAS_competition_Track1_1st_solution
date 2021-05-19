import os, sys
import shutil
import json
import re
import logging
import random
import torch
import pdb

def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)
	print('Experiment dir : {}'.format(path))

	if scripts_to_save is not None:
		os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res

class AvgrageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.avg = 0
		self.sum = 0
		self.cnt = 0

	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt

def load_json(file_path):
	if not os.path.exists(file_path):
		raise RuntimeError('json path {} is not existed.'.format(file_path))

	with open(file_path, 'r') as f:
		model_pool = json.load(f)
		return model_pool
	return None


def get_iter(s, pattern='[0-9]+'):
    idx = re.findall(pattern, s)
    if len(idx) != 1:
    	print('str contains more than one number (index)')
    return int(idx[-1])

def collect_files(work_path, pattern='population_info_iter_[0-9]+.json'):
    dir_list = os.listdir(work_path)
    file_list = []
    for name in dir_list:
        if re.match(pattern, name):
            file_list.append(name)
    file_list.sort(key=get_iter)
    return file_list

def get_common_logger(log_path=None, log_level=logging.INFO, silence=False):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    assert log_path or (not silence), 'must have one handle for logging'
    if log_path:
        fh = logging.FileHandler(log_path, mode='w')
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(format_str))
        logger.addHandler(fh)
    if not silence:
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(logging.Formatter(format_str))
        logger.addHandler(sh)
    return logger

def load_checkpoint(model, model_path, logging, strict=False):
	if os.path.exists(model_path):
		logging.info('=> loading checkpoint %s' % model_path)
		checkpoint = torch.load(model_path, map_location='cpu')
		state_dict = checkpoint['state_dict']
		
		if list(state_dict.keys())[0].startswith('module.') and (not hasattr(model, 'module')):
			state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
		if checkpoint is not None:
			model.load_state_dict(state_dict, strict=strict)
		logging.info('=> loaded checkpoint %s' % model_path)
		return checkpoint
	else:
		logging.info('model path %s is not exists.' % model_path)
		return None
	return None