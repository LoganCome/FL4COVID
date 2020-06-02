#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, covidx_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNN, CovidNet
from models.Fed import FedAvg
from models.test import test_img
from data_loader.covidxdataset import COVIDxDataset
from models.metric import accuracy
from utils.util import print_stats, print_summary, select_model, select_optimizer, MetricTracker



def test(net_g, data_loader):
    # testing
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    elif args.dataset == 'covidx':
        dataset_train = COVIDxDataset(mode='train', n_classes=args.num_classes, dataset_path=args.root_path,
                                     dim=(224, 224))
        dataset_test = COVIDxDataset(mode='test', n_classes=args.num_classes, dataset_path=args.root_path,
                                   dim=(224, 224))
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
    if args.recover != "none":
        net_glob.load_state_dict(torch.load(args.recover))
    print(net_glob)
    net_glob.train()


    # training
    optimizer = optim.Adam(net_glob.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True)

    list_loss = []
    check_points = [i*50 for i in range(1, 10)]

    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        if (epoch+1) in check_points:
            torch.save(net_glob.state_dict(), './save/nn_{}_{}_{}_ckp{}.pkl'.format(args.dataset, args.model, args.epochs, epoch+1))
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # save result
    np.save('./save/nn_{}_{}_{}_loss.npy'.format(args.dataset, args.model, args.epochs), np.array(list_loss))
    torch.save(net_glob.state_dict(), './save/nn_{}_{}_{}.pkl'.format(args.dataset, args.model, args.epochs))

    
    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('./save/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing

    net_glob.eval()
    acc_train, loss_train = test(net_glob, dataset_train)
    acc_test, loss_test = test(net_glob, dataset_test)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
