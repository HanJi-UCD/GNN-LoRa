#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:53:19 2024

@author: han
"""

import pandas as pd
import torch
import json
from torch.utils.data import DataLoader  
import csv
import time
from models import DNN

# test power comsumption and runtime for GNN methods

model_path = 'results/2024-05-24-03-01-16-PM-DNN'
dataset_root = 'dataset/LoRa/LoRadataset_GT_test_1000.csv'
sample_size = 100

############    Load dataset    ############
df = pd.read_csv(dataset_root, header=None)
x_PL = torch.tensor(df.iloc[:, 0:9].values)
y_SF_Ptx = torch.tensor(df.iloc[:, 9:].values)

with open(model_path + '/statistics.json', 'r') as json_file:
    data = json.load(json_file)
nor_x_PL = (x_PL - data['mean']) / data['std']

nor_x_PL = nor_x_PL.float()
test_dataset = DataLoader(nor_x_PL, batch_size=sample_size, shuffle=False)

############    Load DNN model    ############
model = DNN(input_dim=9, output_dim_SF_class=6, output_dim_Ptx_class=5)
checkpoint = torch.load(model_path+'/final_model.pth')
model.load_state_dict(checkpoint['encoder'])
model.eval() 

############    Test    ############
index = 0
runtime = 0
pred_SF_Ptx = torch.zeros_like(y_SF_Ptx)
for data in test_dataset:
    
    start_time = time.time()
    data_SF, data_Ptx = model(data)
    end_time = time.time()
    runtime += end_time - start_time

    pred_SF = torch.argmax(data_SF, dim=2)
    pred_Ptx = torch.argmax(data_Ptx, dim=2)
    
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
with open('pred_labels_DNN_1000.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(saved_data)
    
print('')
print('*'*50)
print(f"Test Accuracy of SF and Ptx are: {[acc_SF, acc_Ptx]}")
print('Average inference time is %s ms:'%(runtime/(len(y_SF_Ptx))*1000))




