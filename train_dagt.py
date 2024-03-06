import argparse
import os

import random
import progressbar
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (SGD only)')
parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--stacks', type=int, default = 500, help='start epoch')
parser.add_argument('--start_epoch', type=int, default = 0, help='start epoch')

parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '', help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '', help='optimizer name for training resume')

parser.add_argument('--dataset', type=str, default = 'dexycb', help='dataset name: nyu, dexycb..')
parser.add_argument('--dataset_path', type=str, default = '../dataset',  help='dataset path')
parser.add_argument('--protocal', type=str, default = 's0',  help='evaluation setting')

parser.add_argument('--test_path', type=str, default = '../dataset',  help='model name for training resume')

parser.add_argument('--model_name', type=str, default = 'handdagt',  help='')
parser.add_argument('--gpu', type=str, default = '0',  help='gpu')

opt = parser.parse_args()

module = importlib.import_module('network_'+opt.model_name)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

torch.cuda.set_device(opt.main_gpu)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset == 'dexycb':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_'+opt.protocal +'_' + opt.model_name+'_'+str(opt.stacks)+'stacks')
	from dataloader import loader 
	opt.JOINT_NUM = 21
elif opt.dataset == 'ho3d':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_v2_' + opt.model_name+'_'+ str(opt.stacks)+'stacks')
	from dataloader import ho3d_loader 
elif opt.dataset == 'nyu':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+ str(opt.stacks)+'stacks')
	from dataloader import loader 
	opt.JOINT_NUM = 14

def _debug(model):
	model = model.netR_1
	print(model.named_paramters())
try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
if opt.dataset == 'nyu':
	train_data = loader.nyu_loader(opt.dataset_path, 'train', aug_para=[10, 0.2, 180], joint_num=opt.JOINT_NUM)
elif opt.dataset == 'ho3d':
	train_data = ho3d_loader.HO3D('train_all', opt.dataset_path, aug_para=[10, 0.2, 180], dataset_version='v2', center_type='joint_mean' )
elif opt.dataset == 'dexycb' :
	train_data = loader.DexYCBDataset(opt.protocal, 'train', opt.dataset_path, aug_para=[10, 0.2, 180])
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
										shuffle=True, num_workers=int(opt.workers), pin_memory=False)

if opt.dataset == 'dexycb' :
	test_data = loader.DexYCBDataset(opt.protocal, 'test', opt.dataset_path)
elif opt.dataset == 'ho3d':
	test_data = ho3d_loader.HO3D('test', opt.dataset_path, dataset_version='v2', center_type='joint_mean' )
elif opt.dataset == 'nyu':
	test_data = loader.nyu_loader(opt.dataset_path, 'test', joint_num=opt.JOINT_NUM)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
model = getattr(module, 'HandModel')(joints=opt.JOINT_NUM, stacks=opt.stacks)

if opt.ngpu > 1:
	model.netR_1 = torch.nn.DataParallel(model.netR_1, range(opt.ngpu))
	model.netR_2 = torch.nn.DataParallel(model.netR_2, range(opt.ngpu))
	model.netR_3 = torch.nn.DataParallel(model.netR_3, range(opt.ngpu))
if opt.model != '':
	model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
	
model.cuda()
# print(model)

parameters = model.parameters()

optimizer = optim.AdamW(parameters, lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06, weight_decay=opt.weight_decay)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
if opt.dataset == 'dexycb':
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
if opt.dataset == 'ho3d':
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

test_best_error = np.inf

# 3. Training and testing
for epoch in range(opt.start_epoch, opt.nepoch):
	scheduler.step(epoch)
	if opt.dataset == 'msra':
		print('======>>>>> Online epoch: #%d, Test: %s, lr=%f  <<<<<======' %(epoch, subject_names[opt.test_index], scheduler.get_lr()[0]))
	else:
		print('======>>>>> Online epoch: #%d, lr=%f  <<<<<======' %(epoch, scheduler.get_lr()[0]))

	# 3.1 switch to train mode
	torch.cuda.synchronize()
	model.train()
	train_mse = 0.0
	train_mse_wld = 0.0
	timer = time.time()

	for i, data in enumerate(tqdm(train_dataloader, ncols=50)):
		
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()       
		# 3.1.1 load inputs and targets

		if opt.dataset == "nyu":
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length = data
		else:
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para = data
		points, gt_xyz, img = points.cuda(),  gt_xyz.cuda(), img.cuda()
		center, M, cube, cam_para = center.cuda(), M.cuda(), cube.cuda(), cam_para.cuda()

		# 3.1.2 compute output
		optimizer.zero_grad()
		loss = model.get_loss(points.transpose(1,2), points.transpose(1,2), img, train_data, center, M, cube, cam_para, gt_xyz.transpose(1,2))

		# 3.1.3 compute gradient and do SGD step
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
		# torch.nn.utils.clip_grad_norm_(model.gru.parameters(), 0.25)
		optimizer.step()
		torch.cuda.synchronize()
		
		# 3.1.4 update training error
		train_mse = train_mse + loss.item()*len(points)

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

	# print mse
	train_mse = train_mse / len(train_data)

	print('mean-square error of 1 sample: %f, #train_data = %d' %(train_mse, len(train_data)))

	if (epoch % 10) == 0:
		torch.save(model.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
		torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))

	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	model.eval()
	test_mse = 0.0
	test_wld_err = 0.0
	timer = time.time()
	for i, data in enumerate(tqdm(test_dataloader, ncols=50)):
		torch.cuda.synchronize()
		with torch.no_grad():
			# 3.2.1 load inputs and targets

			if opt.dataset == "nyu":
				img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length = data
				volume_length = volume_length.cuda()
			else:
				img, points, gt_xyz, uvd_gt, center, M, cube, cam_para = data
				volume_length = 250.
			points, gt_xyz, img = points.cuda(),  gt_xyz.cuda(), img.cuda()
			center, M, cube, cam_para = center.cuda(), M.cuda(), cube.cuda(), cam_para.cuda()


			estimation = model(points.transpose(1,2), points.transpose(1,2), img, test_data, center, M, cube, cam_para)
			loss = model.get_loss(points.transpose(1,2), points.transpose(1,2), img, test_data, center, M, cube, cam_para, gt_xyz.transpose(1,2))

		torch.cuda.synchronize()
		test_mse = test_mse + loss.item()*len(points)

		# 3.2.3 compute error in world cs        
		outputs_xyz = estimation.transpose(1,2)
		diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
		diff_sum = torch.sum(diff,2)
		diff_sum_sqrt = torch.sqrt(diff_sum)
		if opt.dataset == 'nyu' and opt.JOINT_NUM !=14:
			diff_sum_sqrt = diff_sum_sqrt[:, calculate]
		diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
		diff_mean_wld = torch.mul(diff_mean,volume_length.view(-1, 1) / 2 if opt.dataset == "nyu" else 250./2)
		test_wld_err = test_wld_err + diff_mean_wld.sum().item()

	if test_best_error > test_wld_err:
		test_best_error = test_wld_err
		torch.save(model.state_dict(), '%s/best_model.pth' % (save_dir))
		torch.save(optimizer.state_dict(), '%s/best_optimizer.pth' % (save_dir))
				
	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(test_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
	# print mse
	test_mse = test_mse / len(test_data)
	print('mean-square error of 1 sample: %f, #test_data = %d' %(test_mse, len(test_data)))
	test_wld_err = test_wld_err / len(test_data)
	print('average estimation error in world coordinate system: %f (mm)' %(test_wld_err))
	# log
	logging.info('Epoch#%d: train error=%e, train wld error = %f mm, test error=%e, test wld error = %f mm, best wld error = %f' %(epoch, train_mse, train_mse_wld, test_mse, test_wld_err, test_best_error / len(test_data)))