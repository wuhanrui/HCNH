import numpy as np
from utils import load_data, dotdict, seed_everything, accuracy, normalize_sparse_hypergraph_symmetric
import sys
import time
from model import HCNH
import torch
import torch.nn.functional as F
from torch import nn, optim
import os
import argparse


"""
run: python run.py --gpu_id 0 --dataname citeseer
"""



def training(data, args, s = 2021):

    seed_everything(seed = s)

    H_trainX = torch.from_numpy(data.H_trainX.toarray()).float().cuda()
    H_trainX_norm = torch.from_numpy(data.H_trainX_norm.toarray()).float().to_sparse().cuda()
    H_trainY_norm = torch.from_numpy(data.H_trainY_norm.toarray()).float().to_sparse().cuda()
    X = torch.from_numpy(data.X.toarray()).float().to_sparse().cuda()
    Y = torch.from_numpy(data.Y.toarray()).float().to_sparse().cuda()
    
    idx_train = torch.LongTensor(data.idx_train).cuda()
    idx_val = torch.LongTensor(data.idx_val).cuda()
    idx_test = torch.LongTensor(data.idx_test).cuda()
    labels = torch.LongTensor(np.where(data.labels)[1]).cuda()


    epochs = args.epochs
    gamma = args.gamma
    learning_rate = args.learning_rate


    model = HCNH(X.shape[1], Y.shape[1], args.dim_hidden, data.n_class)
    model.cuda()

    criteon = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    
    cost_val = []
    for epoch in range(epochs):
        t = time.time()
        model.train()
        recovered, output, x, y = model(X, H_trainX_norm, Y, H_trainY_norm)
        loss1 = F.nll_loss(output[idx_train], labels[idx_train])
        loss2 = criteon(recovered, H_trainX)
        
        loss_train = loss1 + gamma * loss2
        acc_train = accuracy(output[idx_train], labels[idx_train])
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        cur_loss = loss_train.item()
        
        
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]) + gamma * loss2
        cost_val.append(loss_val.item())
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        if epoch > args.early_stop and cost_val[-1] > np.mean(cost_val[-(args.early_stop+1):-1]):
            print("Early stopping...")
            break
        

#         print('Epoch: {:04d}'.format(epoch+1),
#               'loss_train: {:.4f}'.format(loss_train.item()),
#               'acc_train: {:.4f}'.format(acc_train.item()),
#               'loss_val: {:.4f}'.format(loss_val.item()),
#               'acc_val: {:.4f}'.format(acc_val.item()),
#               'time: {:.4f}s'.format(time.time() - t))
            
        
    # Test
    with torch.no_grad():

        model.eval()
        recovered, output, x, y = model(X, H_trainX_norm, Y, H_trainY_norm)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:", "acc_val: {:.4f}".format(acc_val.item()), "acc_test: {:.4f}".format(acc_test.item()),)
        
    return acc_test.item()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='HCNH')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataname', type=str, nargs='?', default='citeseer', help="dataname to run")
    setting = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.cuda.current_device()
    
    H, X, Y, labels, idx_train_list, idx_val_list = load_data(setting.dataname)
    
    H_trainX = H.copy()
    H_trainY = H_trainX.transpose()
    
    H_trainX_norm = normalize_sparse_hypergraph_symmetric(H_trainX)
    H_trainY_norm = normalize_sparse_hypergraph_symmetric(H_trainY)

    acc_test_list = []
    for trial in range(len(idx_train_list)):
        idx_train = idx_train_list[trial]
        idx_val = idx_val_list[trial]
        idx_test = np.copy(idx_val)
    
        data = dotdict()
        args = dotdict()

        data.X = X
        data.Y = Y
        data.H_trainX = H_trainX
        data.H_trainX_norm = H_trainX_norm
        data.H_trainY_norm = H_trainY_norm
        data.labels = labels
        data.idx_train = idx_train
        data.idx_val = idx_val
        data.idx_test = idx_test
        data.n_class = labels.shape[1]


        dim_hidden = 128
        epochs = 300
        early = 100
        seed = 2021
        
        if setting.dataname == 'pubmed':            
            learning_rate = 0.05
            gamma = 0.01
            weight_decay = 5e-5
        elif setting.dataname == 'citeseer':
            learning_rate = 0.001
            gamma = 0.1
            weight_decay = 0.001
        elif setting.dataname == 'cora':
            learning_rate = 0.005
            gamma = 0.01
            weight_decay = 0.0001

        args.dim_hidden = dim_hidden
        args.weight_decay = weight_decay
        args.epochs = epochs
        args.early_stop = early
        args.learning_rate = learning_rate
        args.gamma = gamma

        acc_test = training(data, args, s=seed)
        acc_test_list.append(acc_test)


    acc_test_list = np.array(acc_test_list) * 100
    m_acc = np.mean(acc_test_list)
    s_acc = np.std(acc_test_list)
    print("Test set results:", "accuracy: {:.4f}({:.4f})".format(m_acc, s_acc))

    
    
    
    
    