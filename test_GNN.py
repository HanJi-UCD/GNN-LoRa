#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:19:57 2024

@author: han
"""

import pandas as pd
import torch
import json
from torch_geometric.data import DataLoader
import utils
import torch.nn as nn
import csv
import time
from models import GraphSAGE
import warnings
warnings.filterwarnings("ignore")

# test power comsumption and runtime for GNN methods

model_path = 'results/2024-05-28-07-36-44-PM-GraphSAGE'
dataset_root = 'dataset/LoRa/LoRadataset_GT_test_1000.csv'
sample_size = 100

############    Load dataset    ############
df = pd.read_csv(dataset_root, header=None)
x_PL = torch.tensor(df.iloc[:, 0:9].values)
y_SF_Ptx = torch.tensor(df.iloc[:, 9:].values)

with open(model_path + '/statistics.json', 'r') as json_file:
    data = json.load(json_file)
nor_x_PL = (x_PL - data['mean']) / data['std']

############    Embedding into graph data    ############
embedding_layer = nn.Linear(1, 11)
embedding_layer.load_state_dict(torch.load('embedding_layer.pth'))

embedded_PL_node = utils.map_to_embedding(nor_x_PL, 11, embedding_layer) # linear embedding for PL, dim: 1 ---> 11
zero_node = torch.zeros(len(nor_x_PL), 1, 11)  # create another empty node: (sample_num, 1, 11)
embedded_PL_node = torch.cat([zero_node, embedded_PL_node], dim=1)
edge_index = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], 
                           [0, 1, 0, 1, 1, 1, 1, 2, 2]], dtype=torch.long)
graph_data_list = utils.create_graph_data(embedded_PL_node, edge_index)
test_dataset = DataLoader(graph_data_list, batch_size=sample_size, shuffle=False)

############    Load GNN model    ############
model = GraphSAGE(feature=11, hidden=64, classes=11)
checkpoint = torch.load(model_path+'/final_model.pth')
model.load_state_dict(checkpoint['encoder'])
model.eval() 


############    Test    ############
index = 0
runtime = 0
pred_SF_Ptx = torch.zeros_like(y_SF_Ptx)
for data in test_dataset:
    
    start_time = time.time()
    data_SF, data_Ptx = model(data, 6)
    end_time = time.time()
    runtime += end_time - start_time
    data_SF = data_SF.argmax(dim=1)
    data_Ptx = data_Ptx.argmax(dim=1)
    
    # data_SF = F.softmax(data_SF, dim=1).argmax(dim=1)
    # data_Ptx = F.softmax(data_Ptx, dim=1).argmax(dim=1)
    pred_SF, pred_Ptx = utils.remove_empty_node(data_SF, data_Ptx, sample_size)
    
    pred_SF = pred_SF.reshape([sample_size, 9]) + 7
    pred_Ptx = (pred_Ptx * 3 + 8).reshape([sample_size, 9])
    
    pred_SF_Ptx[index*sample_size:(index+1)*sample_size, 0:9]  = pred_SF
    pred_SF_Ptx[index*sample_size:(index+1)*sample_size, 9:] = pred_Ptx
    
    index += 1
    
# calculate accuracy
count_SF = torch.sum((pred_SF_Ptx[:, 0:9] != y_SF_Ptx[:, 0:9])).item()
count_Ptx = torch.sum((pred_SF_Ptx[:, 9:] != y_SF_Ptx[:, 9:])).item()
acc_SF = 1 - count_SF/len(y_SF_Ptx)/9;
acc_Ptx = 1 - count_Ptx/len(y_SF_Ptx)/9;
    
# save predicted labels
saved_data = torch.cat([x_PL, pred_SF_Ptx], dim=1)
saved_data = saved_data.numpy()
with open('pred_labels_GNN_1000.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(saved_data)
    
print('')
print('*'*50)
print(f"Test Accuracy of SF and Ptx are: {[acc_SF, acc_Ptx]}")
print('Average inference time is %s ms:'%(runtime/(len(y_SF_Ptx))*1000))
    
    
    
    
    
    
    
    
    
    
    


