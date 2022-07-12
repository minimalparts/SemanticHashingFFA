# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:37:45 2018

@author: akash
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from modules import *
import argparse
from sklearn.metrics import f1_score, accuracy_score
from utils import append_in_tsv, data_loader
from pathlib import Path
import os

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from codecarbon import EmissionsTracker

def timeSince(since):
    now = time.time()
    s = now - since
    #m = math.floor(s / 60)
    #s -= m * 60
    return s

# def accuracy(y_true, y_pred):
#     """
#     example_based_accuracy
#     """
    
#     # compute true positives using the logical AND operator
#     numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1)

#     # compute true_positive + false negatives + false positive using the logical OR operator
#     denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
#     instance_accuracy = numerator/denominator

#     avg_accuracy = np.mean(instance_accuracy)
#     # print(avg_accuracy)
#     return avg_accuracy


def get_tptnfpfn(target, pred):

    tp, tn, fp, fn = 0, 0, 0, 0
    for real_line, pred_line in zip(target, pred):
        for real, pred in zip(real_line, pred_line):
            if pred == 1:
                if real == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if real == 1:
                    fn += 1
                else:
                    tn += 1
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def accuracy(target, pred):
    data = get_tptnfpfn(target, pred)
    sum_ = (data['tp'] + data['tn'] + data['fp'] + data['fn'])
    acc = float(data['tp'] + data['tn']) / sum_
    return acc


def get_f_score(data, beta=1):
    sum_ = (1 + beta**2) * data['tp'] + beta**2 * data['fn'] + data['fp']
    return float(1 + beta**2) * data['tp'] / sum_


def train(args, epoch, model, train_loader, optimizer):

    criterion = args.criterion

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def evaluate(args, model, val_loader):
    criterion=args.criterion
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            if args.dataset != 'tmc' and args.dataset != 'reuters':
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += float(pred.eq(target.data.view_as(pred)).cpu().sum())
                total += float(target.size(0))
            else:
                outputs = torch.sigmoid(output).cpu()     #<--- since you use BCEWithLogitsLoss
                pred = np.round(outputs)
                correct+=accuracy(np.array(target), np.array(pred))
                total += 1
    
    acc=100*(correct / total )
    val_loss /= len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(val_loader.dataset),
        acc))
    return acc, val_loss


def test(args, model, test_loader):
    criterion=args.criterion
    model.eval()
    test_loss, total, correct = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            if args.dataset != 'tmc' and args.dataset != 'reuters':
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += float(pred.eq(target.data.view_as(pred)).cpu().sum())
                total += float(target.size(0))
            else:
                outputs = torch.sigmoid(output).cpu()        #<--- since you use BCEWithLogitsLoss
                pred = np.round(outputs)
                correct+=accuracy(np.array(target), np.array(pred))
                total += 1

    acc=100*(correct / total )
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc


def main_optimizer(lr, neurons_hl):

    # early_stopping = EarlyStopping(patience=5, verbose=True)
    patience=8
    args.lr=lr
    args.neurons_hl=neurons_hl

    model = Model(num_labels, dim_features, neurons_hl)  #(dim_features, num_labels, args.neurons_hl, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    if args.cuda:
        #torch.cuda.set_device(3)
        model.cuda()

    start = time.time()
    time_graph=[]
    e=[]
    best_acc, es = 0, 0
    Path(f'./models/{args.dataset}').mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        e.append(epoch)
        train(args, epoch, model, train_loader, optimizer) 
        val_acc, val_loss=evaluate(args, model, val_loader) 
        seco=timeSince(start)
        time_graph.append(seco)
        
        if val_acc > best_acc:
            best_acc = val_acc
            args.epochs=epoch
            args.val_acc=best_acc
            es = 0                
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, f'./models/{args.dataset}/checkpoint_{lr}_{neurons_hl}.pt')
        else:
            es += 1
            print(f"Counter {es} of {patience+1}")
            if es > patience:
                print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc, "...")
                break

    return best_acc  #it's the val acc for the bayes_opt function

def optimize_bnn():
    def _bnn(lr, neurons_hl):
        neurons_hl = round(neurons_hl)
        print(f'--- lr {lr}, neurons_hl {neurons_hl} ')
        return main_optimizer(lr, neurons_hl)

    optimizer = BayesianOptimization(
        f=_bnn,
        pbounds={"lr": (0.0002, 0.01), "neurons_hl": (64, 512)},
        #random_state=1234,
        verbose=2
    )

    Path(f"./log/{args.dataset}").mkdir(parents=True, exist_ok=True)
    tmp_log_path = f'./log/{args.dataset}/logs.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=3, n_iter=10)
    print("Final result:", optimizer.max)

    params = optimizer.max['params']
    args.neurons_hl=round(params['neurons_hl'])
    args.lr=params['lr']

    checkpoint = torch.load(f'./models/{args.dataset}/checkpoint_{args.lr}_{args.neurons_hl}.pt')
    model = Model(num_labels, dim_features, args.neurons_hl)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_acc=test(args, model, test_loader)
    args.test_acc=test_acc

    if not os.path.exists(f'./models/args.tsv'):
        append_in_tsv(f'./models/args.tsv', list(vars(args).keys()))
    append_in_tsv(f'./models/args.tsv', list(vars(args).values()))

def main(lr, neurons_hl):

    # early_stopping = EarlyStopping(patience=5, verbose=True)
    patience=8
    args.lr=lr
    args.neurons_hl=neurons_hl

    model = Model(num_labels, dim_features, neurons_hl)  #(dim_features, num_labels, args.neurons_hl, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    if args.cuda:
        #torch.cuda.set_device(3)
        model.cuda()

    start = time.time()
    time_graph=[]
    e=[]
    best_acc, es = 0, 0

    if args.dim_reduction:
        f_save_model=f'./models/{args.dataset}/checkpoint_{lr}_{neurons_hl}_umap.pt'
    else:
        f_save_model=f'./models/{args.dataset}/checkpoint_{lr}_{neurons_hl}.pt'

    Path(f'./models/{args.dataset}').mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        e.append(epoch)
        train(args, epoch, model, train_loader, optimizer) 
        val_acc, val_loss=evaluate(args, model, val_loader) 
        seco=timeSince(start)
        time_graph.append(seco)
        
        if val_acc > best_acc:
            best_acc = val_acc
            args.epochs=epoch
            args.val_acc=best_acc
            es = 0                
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, f_save_model)
        else:
            es += 1
            print(f"Counter {es} of {patience+1}")
            if es > patience:
                print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc, "...")
                break
    
    test_acc=test(args, model, test_loader)
    args.test_acc=test_acc

    f_save=f'./models/args_small_models.tsv'
    if not os.path.exists(f_save):
        append_in_tsv(f_save, list(vars(args).keys()))
    append_in_tsv(f_save, list(vars(args).values()))



if __name__ == '__main__':

    tracker = EmissionsTracker(output_dir="./emission_tracking/wiki", project_name="wiki bnn")
    tracker.start()

    parser = argparse.ArgumentParser(description='Binarized weights')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',help='input batch size , default =64')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='N',help='input batch size for testing default=64')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed,default=1)')
    parser.add_argument('--eps', type=float, default=1e-5, metavar='LR',help='learning rate,default=1e-5')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',help='for printing  training data is log interval')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--neurons_hl", type=int, default=128)
    parser.add_argument("--dim_reduction", type=str)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.dataset == "wiki":
        args.train_path="../datasets/wikipedia/wikipedia-train.sp"
        args.spm_model = "../spm/wiki/spm.wiki.model"
        args.spm_vocab = "../spm/wiki/spm.wiki.vocab"
        args.logprob_power = 5
        args.criterion= nn.CrossEntropyLoss()  #nn.BCEWithLogitsLoss()
    if args.dataset == "20news":
        args.train_path="../datasets/20news-bydate/20news-bydate-train.sp"
        args.spm_model = "../spm/20news/spm.20news.model"
        args.spm_vocab = "../spm/20news/spm.20news.vocab"
        args.logprob_power = 4
        args.criterion= nn.CrossEntropyLoss()  #nn.BCEWithLogitsLoss()
    if args.dataset == "wos":
        args.train_path="../datasets/wos/wos11967-train.sp"
        args.spm_model = "../spm/wos/spm.wos.model"
        args.spm_vocab = "../spm/wos/spm.wos.vocab"
        args.logprob_power = 4
        args.criterion= nn.CrossEntropyLoss()  #nn.BCEWithLogitsLoss()
    if args.dataset == "reuters" or args.dataset == "tmc" or args.dataset=="agnews":
        args.train_path=f"../datasets/{args.dataset}/{args.dataset}-train.sp"
        args.spm_model = f"../spm/{args.dataset}/spm.{args.dataset}.model"
        args.spm_vocab = f"../spm/{args.dataset}/spm.{args.dataset}.vocab"
        args.logprob_power = 3
        if args.dataset=='agnews':
            args.criterion=nn.CrossEntropyLoss()  #nn.BCEWithLogitsLoss()
        else:
            args.criterion=nn.BCEWithLogitsLoss()

    train_loader, test_loader, val_loader, num_labels, dim_features = data_loader(args)

    for args.neurons_hl in [32, 64, 128]:
        print('Dataset name:', args.dataset)
        print("num_labels: ", num_labels, "and dim_features: ", dim_features)
        # optimize_bnn()
        main(args.lr, args.neurons_hl)
            
    tracker.stop()
