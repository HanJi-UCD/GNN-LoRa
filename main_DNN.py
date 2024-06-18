#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:40:04 2024

@author: han
"""

import torch
import torch.nn as nn
import utils
from torch.utils.data import TensorDataset, random_split
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
import json
import datetime
import argparse
import warnings
from models import DNN, DNN_single

############    define and revise hyper-parameters    ############
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='Description: GCN-based medel predicts SF and transmitted power in LoRa environment')
parser.add_argument('--epoch-num', default=201, type=int)
parser.add_argument('--learning-rate', default=1e-03, type=float, help='Learning rate')
parser.add_argument('--hidden-dim', default=64, type=float, help='Dim of hidden layer in GCN')
parser.add_argument('--batch-size', default=100, type=int, help='Batch size')
parser.add_argument('--node-num', default=10, type=int, help='10 nodes: 9 nodes have features, 1 node is empty')
parser.add_argument('--edge-num', default=9, type=int, help='directed link numbers')
parser.add_argument('--SF-num', default=6, type=int, help='types(dim) of SF parameter')
parser.add_argument('--Ptx-num', default=5, type=int, help='types(dim) of P_tx parameter')
parser.add_argument('--embedding-dim', default=11, type=int, help='dim of each edge features')
parser.add_argument('--SF-mapping', default={7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5}, type=int, help='one-hot encoding mapping for SF values (7, 8, 9, 10, 11, 12)')
parser.add_argument('--Ptx-mapping', default={8: 0, 11: 1, 14: 2, 17: 3, 20: 4}, type=int, help='one-hot encoding mapping for Ptx values (8, 11, 14, 17, 20)')
parser.add_argument('--test-freq', default=5,help='test training process for each X epochs')
parser.add_argument('--save-dir', default='./results')
parser.add_argument('--dataset_name', default='LoRa')
parser.add_argument('--dataset_root', default='dataset/LoRa/LoRadataset_GT_Linear.csv')
parser.add_argument('--alpha', default='1',type=float)
parser.add_argument('--samples', default='10000',type=int)

args = parser.parse_args()
args.folder_to_save_files = 'results/'+datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p")+'-DNN'
if not os.path.exists(args.folder_to_save_files):
    os.mkdir(args.folder_to_save_files)
    
arg_config_path = os.path.join(args.folder_to_save_files, 'Hyper-parameters.json')
with open (arg_config_path, 'w') as file:
    json.dump(vars(args), file, indent=4)

############    Load dataset    ############
nor_x_PL, y_SF_Ptx = utils.load_dataset(args)

nor_x_PL = nor_x_PL[0:args.samples]
y_SF_Ptx = y_SF_Ptx[0:args.samples]

############    Node and edge features embedding operation  ############
mapped_SF = y_SF_Ptx[:, 0:9] - 7
mapped_Ptx = torch.zeros_like(mapped_SF)
for i in range(args.edge_num):
    mapped_Ptx[:, i] = torch.tensor([args.Ptx_mapping[val.item()] for val in y_SF_Ptx[:, 9+i]])

Y_mapped = torch.cat((mapped_SF, mapped_Ptx), dim=1)

############     Start training         ############ 
dataset = TensorDataset(nor_x_PL, Y_mapped)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# batching for graphs
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

model = DNN(input_dim=args.edge_num, output_dim_SF_class=args.SF_num, output_dim_Ptx_class=args.Ptx_num)
# model = DNN_single(input_dim=args.edge_num, output_dim_class=args.Ptx_num)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

logger = utils.TrainLogger(args.folder_to_save_files)

model.train()
for epoch in range(args.epoch_num):
    training_loss = 0
    ######  training process  ######
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.float()
        labels = labels.long()
        
        outputs_6_class, outputs_5_class = model(inputs)
        #outputs_5_class = model(inputs)
        
        loss1 = 0
        loss2 = 0
        
        for i in range(args.edge_num):
            loss1 += criterion1(outputs_6_class[:, i, :], labels[:, i])
        for i in range(args.edge_num):
            loss2 += criterion2(outputs_5_class[:, i, :], labels[:, i + args.edge_num])
        
        loss = loss1 + args.alpha*loss2
        training_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
    average_train_loss = training_loss / len(train_loader)
    scheduler.step()
    
    ######  test process  ######
    if epoch % args.test_freq == 0:
        model.eval()
        test_loss = 0
        SF_count = 0
        Ptx_count = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.float()
                labels = labels.long()
                
                outputs_6_class, outputs_5_class = model(inputs)
                #outputs_5_class = model(inputs)
                
                loss1 = 0
                loss2 = 0
                
                for i in range(args.edge_num):
                    loss1 += criterion1(outputs_6_class[:, i, :], labels[:, i])
                for i in range(args.edge_num):
                    loss2 += criterion2(outputs_5_class[:, i, :], labels[:, i + args.edge_num])
                
                loss = loss1 + args.alpha*loss2
                test_loss += loss.item()
                
                pre_SF = torch.argmax(outputs_6_class, dim=2)
                pre_Ptx = torch.argmax(outputs_5_class, dim=2)
                SF_label = labels[:, 0:args.edge_num]
                Ptx_label = labels[:, args.edge_num:]
                
                SF_error = torch.sum((SF_label != pre_SF)).item()
                Ptx_error = torch.sum((Ptx_label != pre_Ptx)).item()
                
                SF_count += SF_error
                Ptx_count += Ptx_error
            average_test_loss = test_loss / len(test_loader)
            
        ######   calcualte accuracy   ######
        SF_acc = 1 - SF_count/(args.batch_size*len(test_loader)*args.edge_num);
        Ptx_acc = 1 - Ptx_count/(args.batch_size*len(test_loader)*args.edge_num);
        
        log = [epoch, average_train_loss, average_test_loss, SF_acc, Ptx_acc]
        logger.update(log)
        print('')
        print('*'*50)
        print(f'Learning Rate: {scheduler.get_lr()[0]:.5f}')
        print(f"Epoch {epoch}, Training Loss: {average_train_loss}, Test Loss: {average_test_loss}, SF and Ptx Accuracy: {[SF_acc, Ptx_acc]}")
        print('Running at' + datetime.datetime.now().strftime(" %Y_%m_%d-%I_%M_%S_%p"))

save_model_path = args.folder_to_save_files 
path = save_model_path + '/final_model.pth'
torch.save({'epoch': epoch,
            'encoder':model.state_dict(),}, path)
logger.plot()
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    

