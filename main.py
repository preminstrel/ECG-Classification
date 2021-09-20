import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

## extra packages
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

from quant import *
from fold_batch_norm import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
        
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
        
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
    
class MyDataset(Dataset):
    def __init__(self):
        base_path = './'
        dataset_path =  './Dataset' # Training data
        classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
           'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
        ClassesNum = len(classes)
        X = list()
        y = list()
        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for name in files:
                data_train = scio.loadmat(os.path.join(root, name))# 取出字典里的value
               
              # arr -> list
                data_arr = data_train.get('val')
                data_list = data_arr.tolist()
           
                X.append(data_list[0])
                y.append(int(os.path.basename(root)[0:2]) - 1)  # name -> num
    
        X=np.array(X)
        y=np.array(y)
        X = standardization(X)
        X = X.reshape((1000,1,3600))
        y = y.reshape((1000))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.len = X_train.shape[0] # 取第0元素：长度
        self.x_train = torch.from_numpy(X_train).float().to("cuda")
        self.y_train = torch.from_numpy(y_train).long().to("cuda")
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index] # 返回对应样本即可
    def __len__(self):
        return self.len
    
class TestDataset(Dataset):
    def __init__(self):
        base_path = './'
        dataset_path =  './Dataset' # Training data
        classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
           'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
        ClassesNum = len(classes)
        X = list()
        y = list()
        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for name in files:
                data_train = scio.loadmat(os.path.join(root, name))# 取出字典里的value
               
              # arr -> list
                data_arr = data_train.get('val')
                data_list = data_arr.tolist()
           
                X.append(data_list[0])
                y.append(int(os.path.basename(root)[0:2]) - 1)  # name -> num
    
        X=np.array(X)
        y=np.array(y)
        X = standardization(X)
        X = X.reshape((1000,1,3600))
        y = y.reshape((1000))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.len = X_test.shape[0] # 取第0元素：长度
        self.x_test = torch.from_numpy(X_test).float().to("cuda")
        self.y_test = torch.from_numpy(y_test).long().to("cuda")
    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index] # 返回对应样本即可
    def __len__(self):
        return self.len    
        
class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class arrhythmia_classifier(nn.Module):
    def __init__(self, in_channels=1):
        super(arrhythmia_classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,8,16,stride=2,padding=7),
            nn.ReLU(),
            #nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=8,stride=4),
   
            nn.Conv1d(8,12,12,padding=5,stride=2),
            nn.ReLU(),
            #nn.BatchNorm1d(16),
            nn.MaxPool1d(4,stride=2),
            
            nn.Conv1d(12,32,9,stride=1,padding=4),
            nn.ReLU(),
            nn.MaxPool1d(5,stride=2),
            
            nn.Conv1d(32,64,7,stride=1,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4,stride=2),
            
            nn.Conv1d(64,64,5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            
            nn.Conv1d(64,64,3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            
            nn.Conv1d(64,72,3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            Flatten(),
            nn.Linear(in_features=216, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=64, out_features=17),
        )

    def forward(self, x, ex_features=None):
        return self.cnn(x)


def calc_next_len_conv1d(current_len=112500, kernel_size=16, stride=8, padding=0, dilation=1):
    return int(np.floor((current_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
###===================================================================####
def mystr2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

## define parser  # 定义程序检查机制
parser = argparse.ArgumentParser(description='PyTorch PWLQ code on ECGNet')
parser.add_argument('-data', default='/home/preminstrel/ECG-Classification')
parser.add_argument('-a', '--arch', default='test_1', help='network architecture')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-p', '--print-freq', default=100, type=int)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

## extras for quantization  额外的量化
parser.add_argument('-fbn', '--fold_bn', dest='fold_bn', action='store_true',
                    help='fold batch normalization')
parser.add_argument('-quant', '--quantize', dest='quantize', action='store_true',
                    help='quantize model')
parser.add_argument('-gs', '--get_stats', dest='get_stats', action='store_true',
                    help='get stats of activations')
parser.add_argument('-wb', '--wei_bits', '--weight-bits', default=0.0, type=float,
                    metavar='WB', help='weight quantization bits')
parser.add_argument('-ab', '--act_bits', '--activation-bits', default=0.0, type=float,
                    metavar='AB', help='activation quantization bits')
parser.add_argument('-sb', '--scale_bits', default=0.0, type=float,
                    metavar='SB', help='scale/shift quantization bits')
parser.add_argument('-wq', '--wei_quant_scheme', default='none', type=str,
                    choices=['uniform', 'pw-2', 'pw-1'],
                    help='weight quantization scheme: uniform, PWLQ')
parser.add_argument('-aq', '--act_clip_method', default='top_10', type=str,
                    help='activations clip-quantization method'
                    'choices: none, on-the-fly, clip_1.0, top_10, etc.')
parser.add_argument('-bc', '--bias_corr', default=False, type=mystr2bool,
                    help='Whether to use bias correction for weights quantization')
parser.add_argument('-appx', '--approximate', default=False, type=mystr2bool,
                    help='Whether to use approximated optimal breakpoint')
parser.add_argument('-bkp', '--break-point', default='none', type=str,
                    help='how to get optimal breakpoint: norm, laplace, search')
parser.add_argument('-sr', '--save_res', default=True, type=mystr2bool,
                    help='save results')
parser.add_argument('-cms', '--comments', default='', type=str,
                    help='make comments')

## main function Main函数
def main(model):



    ### =======================================================   ##
    best_acc1 = 0
    total_start_time = time.time()
    
    args = parser.parse_args()
    print(str(args))
    print()

    # use one GPU to get the activation stats 
    if args.get_stats:
        args.gpu = 0
        args.batch_size = 4

    if args.gpu is not None:
        print("Use GPU: {} for the calibration of activation ranges".format(args.gpu))

    # load pre-trained model 
    print("=> using pre-trained model '{}'".format(args.arch))
    #model = models.__dict__[args.arch](pretrained=True)
    checkpoint = model.state_dict()
    #print(model.state_dict())
    print('----- pretrained model loaded -----')

    ## fold batch normalization
    if args.fold_bn:
        checkpoint, weight_layers = fold_batch_norm(checkpoint, arch=args.arch)

    # quantize weights
    rmse = 0
    if args.quantize:
        print('quantize weights ...')
        assert(args.fold_bn)
        checkpoint, rmse = quant_checkpoint(checkpoint, weight_layers, args)
    
    # load the updated weights
    model.load_state_dict(checkpoint)
    del checkpoint

    # quantize or load activation stats
    model = quant_model_acts(model, args.act_bits, args.get_stats, args.batch_size)
    if args.quantize and not args.get_stats:
        act_stats_save_path = 'stats/%s_act_stats.pth' % args.arch
        mode = load_model_act_stats(model, act_stats_save_path, args.act_clip_method)

    # use GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    cudnn.benchmark = True

    # load data ================================ 加载数据===============================================>
#     crop_size = 224
#     scale = 0.875
#     if args.arch.startswith('inception'):
#         crop_size = 299
#     large_crop_size = int(round(crop_size / scale))
#     print('\nlarger crop size: ', large_crop_size)
#     print('center crop size: ', crop_size)

#     traindir = os.path.join(args.data, 'train')
#     valdir = os.path.join(args.data, 'val')
#     shuffle_option = False
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
    
#     if args.get_stats:
#         valdir = os.path.join(args.data, 'train')
#         shuffle_option = True

#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#             transforms.Resize(large_crop_size),
#             transforms.CenterCrop(crop_size),
#             transforms.ToTensor(),
#             normalize,
#         ])),
#         batch_size=args.batch_size, shuffle=shuffle_option,
#         num_workers=8, pin_memory=True)
    
    # get activation stats on training data
    if args.get_stats:
        # calibrate the activation ranges
        validate(test_loader, model, criterion, args)

        # save the activation stats
        os.makedirs('stats/', exist_ok=True)
        act_stats_save_path = 'stats/%s_act_stats.pth' % args.arch
        save_model_act_stats(model, act_stats_save_path)

        return

    # evaluate on validation dataset
    val_start_time = time.time()
    top1_avg_acc, top5_avg_acc = validate(test_loader, model, criterion, args)
    print('\nvalidation time: %.2f min' % ((time.time() - val_start_time) / 60))

    # save accuracy results
    save_acc_res = True
    save_comments = args.comments 
    if save_acc_res:
        os.makedirs('results/', exist_ok=True)
        table_path = 'results/accuracy_results_%s.csv' % args.arch 

        new_df = pd.DataFrame({'model': [args.arch], 'quantize': [args.quantize], 
            'wei_bits': [args.wei_bits], 'wei_quant_scheme': [args.wei_quant_scheme],  
            'bias_corr': ['BC: yes' if args.bias_corr else 'BC: no'], 
            'approximate': ['appx: yes' if args.approximate else 'appx: no'], 
            'scale_bits': [args.scale_bits],
            'act_bits': [args.act_bits], 'act_clip_method': [args.act_clip_method],
            'break_point': [args.break_point], 
            'wei_quant_rmse': [rmse], 
            'top1': [float(top1_avg_acc)], 'top5': [float(top5_avg_acc)], 
            'comments': [save_comments], 
            'time': [( datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 
                '%.2f min' % ((time.time() - total_start_time)/60) )]})
        
        new_df = new_df[['model', 'quantize', 'act_bits', 'wei_bits', 'scale_bits', 'act_clip_method', 
                        'wei_quant_scheme', 'break_point', 'approximate', 'bias_corr',  
                        'wei_quant_rmse', 'top1', 'top5', 'time', 'comments']]
        
        if os.path.exists(table_path):
            old_df = pd.read_csv(table_path)
            new_df = old_df.append(new_df)
        new_df.to_csv(table_path, index = False)
        
    return

# 准确率
def validate(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            if args.get_stats and (i + 1) * args.batch_size >= 512:
                break

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    base_path = './'
    dataset_path =  './Dataset' # Training data
    classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW','PVC', 'Bigeminy', 'Trigeminy', 
           'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
    ClassesNum = len(classes)
    X = list()
    y = list()
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            data_train = scio.loadmat(os.path.join(root, name))# 取出字典里的value
            
           # arr -> list
            data_arr = data_train.get('val')
            data_list = data_arr.tolist()
        
            X.append(data_list[0])
            y.append(int(os.path.basename(root)[0:2]) - 1)  # name -> num

    X=np.array(X)
    y=np.array(y)
    X = standardization(X)
    X = X.reshape((1000,1,3600))
    y = y.reshape((1000))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("X_train : ", len(X_train))
    print("X_test  : ", len(X_test))
    print("shape of X_train : ", np.shape(X_train[0]))
    print("shape of y_train : ", np.shape(y_train))
    print("shape of X_test : ", np.shape(X_test))
    print("shape of y_test : ", np.shape(y_test))

    batch_size = 16   
    train_dataset = MyDataset()
    test_dataset = TestDataset()
    train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=0)


    in_channels_ = 1
    num_segments_in_record = 100
    segment_len = 3600   # 3600 采样
    num_classes = 17
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = arrhythmia_classifier().to(device)
    from torchsummary import summary
    #summary(model, input_size=(1, 3600))
    model = torch.load('test_1.pt',map_location='cuda')

    start_time = time.time()
    main(model)
    end_time = time.time()
    print('total running time: %.2f min\n' % ((end_time - start_time) / 60))
