import argparse
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import math
from torchvision import datasets, transforms
import numpy as np
from binarynet import ConvLayer_bin, FCLayer_bin
from myoptimizer import ALQ_optimizer
from train import get_accuracy, train_fullprecision, train_basis, train_basis_STE, train_coordinate, validate, test, prune, initialize, save_model, save_model_ori, save_model_simple

# Defining the network (ECGNet5)
in_channels_ = 1
num_segments_in_record = 100
segment_len = 3600   # 3600 采样
num_classes = 17


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


class ECGNet(nn.Module):
    def __init__(self, in_channels=in_channels_):
        super(ECGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 16, stride=2, padding=7),
            nn.ReLU(),
            # nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=8, stride=4),

            nn.Conv1d(8, 12, 12, padding=5, stride=2),
            nn.ReLU(),
            # nn.BatchNorm1d(16),
            nn.MaxPool1d(4, stride=2),

            nn.Conv1d(12, 32, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),

            nn.Conv1d(32, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=2),

            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 72, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.classifier = torch.nn.Sequential(
            nn.Linear(in_features=216, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=64, out_features=17),
        )

    def forward(self, x, ex_features=None):
        x = self.features(x)
        x = x.view((-1, 216))
        x = self.classifier(x)
        return x


class MyDataset(Dataset):
    def __init__(self):
        base_path = './'
        dataset_path = './Dataset'
        classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW', 'PVC', 'Bigeminy',
                   'Trigeminy', 'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
        ClassesNum = len(classes)
        X = list()
        y = list()

        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for name in files:
                data_train = scio.loadmat(
                    os.path.join(root, name))  # 取出字典里的value

        # arr -> list
                data_arr = data_train.get('val')
                data_list = data_arr.tolist()
                X.append(data_list[0])  # [[……]] -> [ ]
                y.append(int(os.path.basename(root)[0:2]) - 1)

        X = np.array(X)
        y = np.array(y)
        X = standardization(X)
        X = X.reshape((1000, 1, 3600))
        y = y.reshape((1000))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        #print("X_train : ", len(X_train))
        #print("shape of X_train : ", np.shape(X_train[0]))
        #print("shape of y_train : ", np.shape(y_train))
        self.len = X_train.shape[0]  # 取第0元素：长度
        self.x_train = torch.from_numpy(X_train).float().to("cuda")
        self.y_train = torch.from_numpy(y_train).long().to("cuda")

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]  # 返回对应样本即可

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self):
        base_path = './'
        dataset_path = './Dataset'
        classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW', 'PVC', 'Bigeminy',
                   'Trigeminy', 'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
        ClassesNum = len(classes)
        X = list()
        y = list()

        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for name in files:
                data_train = scio.loadmat(
                    os.path.join(root, name))  # 取出字典里的value

        # arr -> list
                data_arr = data_train.get('val')
                data_list = data_arr.tolist()
                X.append(data_list[0])  # [[……]] -> [ ]
                y.append(int(os.path.basename(root)[0:2]) - 1)

        X = np.array(X)
        y = np.array(y)
        X = standardization(X)
        X = X.reshape((1000, 1, 3600))
        y = y.reshape((1000))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        #print("X_test  : ", len(X_test))
        #print("shape of X_test : ", np.shape(X_test))
        #print("shape of y_test : ", np.shape(y_test))
        self.len = X_test.shape[0]  # 取第0元素：长度
        self.x_test = torch.from_numpy(X_test).float().to("cuda")
        self.y_test = torch.from_numpy(y_test).long().to("cuda")

    def __getitem__(self, index):
        return self.x_test[index], self.y_test[index]  # 返回对应样本即可

    def __len__(self):
        return self.len


class ValDataset(Dataset):
    def __init__(self):
        base_path = './'
        dataset_path = './Dataset'
        classes = ['NSR', 'APB', 'AFL', 'AFIB', 'SVTA', 'WPW', 'PVC', 'Bigeminy',
                   'Trigeminy', 'VT', 'IVR', 'VFL', 'Fusion', 'LBBBB', 'RBBBB', 'SDHB', 'PR']
        ClassesNum = len(classes)
        X = list()
        y = list()

        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for name in files:
                data_train = scio.loadmat(
                    os.path.join(root, name))  # 取出字典里的value

        # arr -> list
                data_arr = data_train.get('val')
                data_list = data_arr.tolist()
                X.append(data_list[0])  # [[……]] -> [ ]
                y.append(int(os.path.basename(root)[0:2]) - 1)

        X = np.array(X)
        y = np.array(y)
        X = standardization(X)
        X = X.reshape((1000, 1, 3600))
        y = y.reshape((1000))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)
        #print("X_val  : ", len(X_val))
        #print("shape of X_val : ", np.shape(X_val))
        #print("shape of y_val : ", np.shape(y_val))
        self.len = X_val.shape[0]  # 取第0元素：长度
        self.x_val = torch.from_numpy(X_val).float().to("cuda")
        self.y_val = torch.from_numpy(y_val).long().to("cuda")

    def __getitem__(self, index):
        return self.x_val[index], self.y_val[index]  # 返回对应样本即可

    def __len__(self):
        return self.len


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data',
                        help='ECG dataset directory')
    parser.add_argument('--val_size', type=int, default=200,
                        help='the number of samples in validation dataset')
    parser.add_argument('--model_ori', type=str, default='./ECGNet_model_ori.pth',
                        help='the file of the original full precision ECGNet model')
    parser.add_argument('--model', type=str, default='./ECGNet_model.pth',
                        help='the file of the quantized ECGNet model')
    parser.add_argument('--PRETRAIN', action='store_true',
                        help='train the original full precision ECGNet model')  # 全精度
    parser.add_argument('--ALQ', action='store_true',
                        help='adaptive loss-aware quantize ECGNet model')       # ALQ
    parser.add_argument('--POSTTRAIN', action='store_true',
                        help='posttrain the final quantized ECGNet model')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--R', type=int, default=15,
                        help='the number of outer iterations, also the number of pruning')
    parser.add_argument('--epoch_prune', type=int, default=1,
                        help='the number of epochs for pruning')
    parser.add_argument('--epoch_basis', type=int, default=8,
                        help='the number of epochs for optimizing bases')
    parser.add_argument('--ld_basis', type=float, default=0.8,
                        help='learning rate decay factor for optimizing bases')
    parser.add_argument('--epoch_coord', type=int, default=10,
                        help='the number of epochs for optimizing coordinates')
    parser.add_argument('--ld_coord', type=float, default=0.8,
                        help='learning rate decay factor for optimizing coordinates')
    parser.add_argument('--wd', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--pr', type=float, default=0.7,    # prune ratio
                        help='the pruning ratio of alpha')
    parser.add_argument('--top_k', type=float, default=0.005,
                        help='the ratio of selected alpha in each layer for resorting')
    parser.add_argument('--structure', type=str, nargs='+', choices=['channelwise', 'kernelwise', 'pixelwise', 'subchannelwise'],
                        default=['kernelwise', 'kernelwise', 'kernelwise', 'kernelwise', 'kernelwise',
                                 'kernelwise', 'kernelwise', 'subchannelwise', 'subchannelwise'],
                        help='the structure-wise used in each layer')
    parser.add_argument('--subc', type=int, nargs='+', default=[0, 0, 0, 0, 0, 0, 0, 2, 1],  # Matters!!
                        help='number of subchannels when using subchannelwise')
    parser.add_argument('--max_bit', type=int, nargs='+', default=[7, 7, 6, 6, 6, 6, 6, 6, 6],
                        help='the maximum bitwidth used in initialization')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='the number of training samples in each batch')
    args = parser.parse_args()

    # Prepare Dataset
    batch_size = 16
    train_dataset = MyDataset()
    test_dataset = TestDataset()
    val_dataset = ValDataset()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    print("Train_loader ready...")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)
    print("Test_loader ready...")
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    # 全精度 FULL
    if args.PRETRAIN:
        print('pretraining...')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = ECGNet().to(device)
        from torchsummary import summary
        summary(net, input_size=(1, 3600))

        # Construct Loss and Optimizer
        loss_func = torch.nn.CrossEntropyLoss().cuda()
        #optimizer = torch.optim.SGD(net.parameters(), lr=5e-2, momentum=0.5)
        optimizer = optim.Adam(net.parameters(
        ), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)

        get_accuracy(net, train_loader, loss_func)
        val_accuracy = validate(net, val_loader, loss_func)
        best_acc = val_accuracy[0]
        test(net, test_loader, loss_func)
        save_model_ori(args.model_ori, net, optimizer)

        for epoch in range(100):
            # if epoch%30 == 0:
            #    optimizer.param_groups[0]['lr'] *= 0.9
            train_fullprecision(net, train_loader, loss_func, optimizer, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0] > best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model_ori(args.model_ori, net, optimizer)

    # ALQ
    if args.ALQ:
        print('adaptive loss-aware quantization...')

        net = ECGNet().cuda()
        loss_func = torch.nn.CrossEntropyLoss().cuda()

        print('loading pretrained full precision ECGNet model ...')
        checkpoint = torch.load(args.model_ori)
        net.load_state_dict(checkpoint['net_state_dict'])
        for name, param in net.named_parameters():
            print(name)
            print(param.size())
        print('initialization (structured sketching)...')
        parameters_w, parameters_b, parameters_w_bin = initialize(
            net, train_loader, loss_func, args.structure, args.subc, args.max_bit)
        optimizer_b = torch.optim.Adam(parameters_b, weight_decay=args.wd)
        optimizer_w = ALQ_optimizer(parameters_w, weight_decay=args.wd)
        val_accuracy = validate(net, val_loader, loss_func)
        best_acc = val_accuracy[0]
        test(net, test_loader, loss_func)
        save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)
        # print(parameters_w)
        # save_model_simple('./ALQ_in.pt',net)
        num_training_sample = len(train_dataset)
        M_p = (args.pr/args.top_k)/(args.epoch_prune *
                                    math.ceil(num_training_sample/args.batch_size))

        for r in range(args.R):

            print('outer iteration: ', r)
            optimizer_b.param_groups[0]['lr'] = args.lr
            optimizer_w.param_groups[0]['lr'] = args.lr

            print('optimizing basis...')
            for q_epoch in range(args.epoch_basis):
                optimizer_b.param_groups[0]['lr'] *= args.ld_basis
                optimizer_w.param_groups[0]['lr'] *= args.ld_basis
                train_basis(net, train_loader, loss_func, optimizer_w,
                            optimizer_b, parameters_w_bin, q_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                if val_accuracy[0] > best_acc:
                    best_acc = val_accuracy[0]
                    test(net, test_loader, loss_func)
                    #save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)

            print('optimizing coordinates...')
            for p_epoch in range(args.epoch_coord):
                optimizer_b.param_groups[0]['lr'] *= args.ld_coord
                optimizer_w.param_groups[0]['lr'] *= args.ld_coord
                train_coordinate(net, train_loader, loss_func,
                                 optimizer_w, optimizer_b, parameters_w_bin, p_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                if val_accuracy[0] > best_acc:
                    best_acc = val_accuracy[0]
                    test(net, test_loader, loss_func)
                    #save_model(args.model, net, optimizer_w, optimizer_b, parameters_w_bin)

            print('pruning...')
            for t_epoch in range(args.epoch_prune):
                prune(net, train_loader, loss_func, optimizer_w,
                      optimizer_b, parameters_w_bin, [args.top_k, M_p], t_epoch)
                val_accuracy = validate(net, val_loader, loss_func)
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net, optimizer_w,
                           optimizer_b, parameters_w_bin)
        save_model_simple('./ECGNet_model_q_afterALQ.pt', net)
        print(net.features[0].weight)
        print(net.features[3].weight)
        print(net.features[6].weight)
        print(net.features[9].weight)
        print(net.features[12].weight)
        print(net.features[15].weight)
        print(net.features[18].weight)
        print(net.classifier[0].weight)
        print(net.classifier[3].weight)
        torch.save(net, 'a.pth')
    if args.POSTTRAIN:
        print('posttraining...')

        net = ECGNet().cuda()
        loss_func = torch.nn.CrossEntropyLoss().cuda()

        parameters_w = []
        parameters_b = []
        for name, param in net.named_parameters():
            if 'weight' in name and param.dim() > 1:
                parameters_w.append(param)
            else:
                parameters_b.append(param)

        optimizer_b = torch.optim.Adam(parameters_b, weight_decay=args.wd)
        optimizer_w = ALQ_optimizer(parameters_w, weight_decay=args.wd)

        print('load quantized ECGNet model...')
        checkpoint = torch.load(args.model)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer_w.load_state_dict(checkpoint['optimizer_w_state_dict'])
        optimizer_b.load_state_dict(checkpoint['optimizer_b_state_dict'])
        for state in optimizer_b.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        for state in optimizer_w.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        num_weight_layer = 0.
        num_bit_layer = 0.
        print('currrent binary filter number per layer: ')
        for p_w_bin in parameters_w_bin:
            print(p_w_bin.num_bin_filter)
        print('currrent average bitwidth per layer: ')
        for p_w_bin in parameters_w_bin:
            num_weight_layer += p_w_bin.num_weight
            num_bit_layer += p_w_bin.avg_bit*p_w_bin.num_weight
            print(p_w_bin.avg_bit)
        print('currrent average bitwidth: ', num_bit_layer/num_weight_layer)

        get_accuracy(net, train_loader, loss_func)
        val_accuracy = validate(net, val_loader, loss_func)
        best_acc = val_accuracy[0]
        test(net, test_loader, loss_func)
        optimizer_b.param_groups[0]['lr'] = args.lr
        optimizer_w.param_groups[0]['lr'] = args.lr

        print('optimizing basis with STE...')
        for epoch in range(50):
            optimizer_b.param_groups[0]['lr'] *= 0.95
            optimizer_w.param_groups[0]['lr'] *= 0.95
            train_basis_STE(net, train_loader, loss_func,
                            optimizer_w, optimizer_b, parameters_w_bin, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0] > best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net, optimizer_w,
                           optimizer_b, parameters_w_bin)

        print('optimizing coordinates...')
        for epoch in range(20):
            optimizer_b.param_groups[0]['lr'] *= 0.9
            optimizer_w.param_groups[0]['lr'] *= 0.9
            train_coordinate(net, train_loader, loss_func,
                             optimizer_w, optimizer_b, parameters_w_bin, epoch)
            val_accuracy = validate(net, val_loader, loss_func)
            if val_accuracy[0] > best_acc:
                best_acc = val_accuracy[0]
                test(net, test_loader, loss_func)
                save_model(args.model, net, optimizer_w,
                           optimizer_b, parameters_w_bin)
        save_model_simple('./ECGNet_model_q.pth', net)
        torch.save(net, './test.pth')
