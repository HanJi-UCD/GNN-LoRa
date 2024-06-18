#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:11:52 2024

@author: han
"""

import torch
import torch.nn as nn
import utils
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
import json
import datetime
import argparse
import warnings
import torch.nn.functional as F
from models import GCN, DirGCNConv, GAT, GraphSAGE

############    define and revise hyper-parameters    ############
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='Description: GNN-based medel predicts SF and transmitted power in LoRa environment')
parser.add_argument('--epoch-num', default=101, type=int)
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
parser.add_argument('--dataset_root', default='dataset/LoRa/LoRadataset_GT_batch2.csv')
parser.add_argument('--model-name', default='GraphSAGE',help='GCN, GAT, GraphSAGE, DirGCN')
parser.add_argument('--ls', default='CE',help='CE, MSE')
parser.add_argument('--alpha1', default='1',type=float)
parser.add_argument('--alpha2', default='1',type=float)

args = parser.parse_args()
args.folder_to_save_files = 'results/'+datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S-%p")+'-%s'%args.model_name
if not os.path.exists(args.folder_to_save_files):
    os.mkdir(args.folder_to_save_files)
    
arg_config_path = os.path.join(args.folder_to_save_files, 'Hyper-parameters.json')
with open (arg_config_path, 'w') as file:
    json.dump(vars(args), file, indent=4)

############    Load dataset    ############
nor_x_PL, y_SF_Ptx = utils.load_dataset(args)


############    Node and edge features embedding operation  ############
embedding_layer = nn.Linear(1, args.embedding_dim)
embedding_layer.load_state_dict(torch.load('embedding_layer.pth'))

embedded_PL_node = utils.map_to_embedding(nor_x_PL, args.embedding_dim, embedding_layer) # linear embedding for PL, dim: 1 ---> 11
zero_node = torch.zeros(len(nor_x_PL), 1, args.embedding_dim)  # create another empty node: (sample_num, 1, 11)
embedded_PL_node = torch.cat([zero_node, embedded_PL_node], dim=1)
embedded_SF_Ptx_label = utils.process_edge_features(y_SF_Ptx, args) # one-hot embedding for labels, dim: 2 ---> 11

edge_index = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], 
                           [0, 1, 0, 1, 1, 1, 1, 2, 2]], dtype=torch.long)

############     Create Graph data      ############  
graph_data_list = utils.create_graph_data(embedded_PL_node, edge_index, embedded_SF_Ptx_label)

############     Plot Graph      ############ 
# utils.draw_graph(graph_data_list[0])

############     Start training         ############ 
split_ratio = 0.8
split_idx = int(len(graph_data_list) * split_ratio)
train_dataset = graph_data_list[:split_idx]
test_dataset = graph_data_list[split_idx:]
    
# batching for graphs
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

############     Initialize GNN models      ############ 
if args.model_name == 'GCN':
    model = GCN(input_dim=args.embedding_dim, hidden_dim=args.hidden_dim, output_dim=args.embedding_dim)
elif args.model_name == 'GAT':
    model = GAT(features=args.embedding_dim, hidden=args.hidden_dim, classes=args.embedding_dim, heads=4)
elif args.model_name == 'GraphSAGE':
    model = GraphSAGE(feature=args.embedding_dim, hidden=args.hidden_dim, classes=args.embedding_dim)
else: # bad performance now
    model = DirGCNConv(input_dim=args.embedding_dim, output_dim=args.embedding_dim, alpha = 1/2)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
if args.ls == 'CE':
    criterion = nn.CrossEntropyLoss()
else: 
    criterion = nn.MSELoss()
logger = utils.TrainLogger(args.folder_to_save_files)

model.train()
for epoch in range(args.epoch_num):
    training_loss = 0
    ######  training process  ######
    for batch_data in train_loader:
        optimizer.zero_grad()
        data_SF, data_Ptx = model(batch_data, args.SF_num)
        if args.ls == 'CE':
            data_SF = F.sigmoid(data_SF)
            data_Ptx = F.sigmoid(data_Ptx)
        else:
            data_SF = F.softmax(data_SF)
            data_Ptx = F.softmax(data_Ptx)

        # convert predicted output results (remove empty node) features on node-level ??? 
        pred_SF, pred_Ptx = utils.remove_empty_node(data_SF, data_Ptx, args.batch_size)
        
        # get labels from one-hot enbeddings
        if args.ls == 'CE': # calssification task
            label_SF = batch_data.y[:, 0:args.SF_num].argmax(dim=1)
            label_Ptx = batch_data.y[:, args.SF_num:].argmax(dim=1)
        else: # regression task
            label_SF = batch_data.y[:, 0:args.SF_num]
            label_Ptx = batch_data.y[:, args.SF_num:]
        
        loss = args.alpha1*criterion(pred_SF, label_SF) + args.alpha2*criterion(pred_Ptx, label_Ptx)
        training_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
    average_train_loss = training_loss / len(train_loader)
    scheduler.step()
    
    ######  test process  ######
    if epoch % args.test_freq == 0:
        model.eval()
        test_loss = 0.0
        SF_count = 0
        Ptx_count = 0
        with torch.no_grad():
            for batch_data in test_loader:
                data_SF, data_Ptx = model(batch_data, args.SF_num)
                if args.ls == 'CE':
                    data_SF = F.sigmoid(data_SF)
                    data_Ptx = F.sigmoid(data_Ptx)
                else: 
                    data_SF = F.softmax(data_SF)
                    data_Ptx = F.softmax(data_Ptx)
                
                # convert predicted output results (remove empty node)
                pred_SF, pred_Ptx = utils.remove_empty_node(data_SF, data_Ptx, args.batch_size)
                # get labels from one-hot enbeddings
                if args.ls == 'CE': # calssification task
                    label_SF = batch_data.y[:, 0:args.SF_num].argmax(dim=1)
                    label_Ptx = batch_data.y[:, args.SF_num:].argmax(dim=1)
                else: # regression task
                    label_SF = batch_data.y[:, 0:args.SF_num]
                    label_Ptx = batch_data.y[:, args.SF_num:]
                    
                loss = args.alpha1*criterion(pred_SF, label_SF) + args.alpha2*criterion(pred_Ptx, label_Ptx)
                test_loss += loss.item()
                
                if args.ls == 'MSE': # calssification task
                    label_SF = label_SF.argmax(dim=1)
                    label_Ptx = label_Ptx.argmax(dim=1)
                
                SF_err, Ptx_err = utils.cal_accuracy(pred_SF, pred_Ptx, label_SF, label_Ptx)
                SF_count += SF_err
                Ptx_count += Ptx_err
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
    
    
    
    
    
    
    
    

