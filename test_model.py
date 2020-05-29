#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, covidx_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNN, CovidNet
from models.Fed import FedAvg
from models.test import test_img
from data_loader.covidxdataset import COVIDxDataset
from models.metric import accuracy
from utils.util import print_stats, print_summary, select_model, select_optimizer, MetricTracker

import os




if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'covidx':
        dataset_train = COVIDxDataset(mode='train', n_classes=args.num_classes, dataset_path=args.root_path,
                                     dim=(224, 224))
        dataset_test = COVIDxDataset(mode='test', n_classes=args.num_classes, dataset_path=args.root_path,
                                   dim=(224, 224))
        '''
        train_params = {'batch_size': args.local_bs,
                        'shuffle': True,
                        'num_workers': 2}

        test_params = {'batch_size': args.bs,
                       'shuffle': False,
                       'num_workers': 1}
        train_generator = DataLoader(dataset_train, **train_params)
        test_generator = DataLoader(dataset_test, **test_params)
        '''

        if args.iid:
            dict_users = covidx_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in COVIDx')
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'covidnet_small':
        net_glob = CovidNet('small', n_classes=args.num_classes).to(args.device)
    elif args.model == 'covidnet_large':
        net_glob = CovidNet('large', n_classes=args.num_classes).to(args.device)
    elif args.model in ['resnet18', 'mobilenet2', 'densenet169', 'resneXt']:
        net_glob = CNN(args.num_classes, args.model).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.load_state_dict(torch.load(args.test))
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    update_list = [0 for i in range(args.num_users)]
    check_point = [i*10 for i in range(1, 10)]
    print('v1.3')


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

