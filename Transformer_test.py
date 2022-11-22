#%%
import torch 
from torch.utils.data import DataLoader, Dataset 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from Utility import *

model = torch.load('C:/Users/USER/Programming/Excavator/Transformer/Trained_model/transformer_1_3lyr.pt')
#print(model)
#%%
test_input_list = make_input_tensor('C:/Users/USER/Programming/Excavator/Data/Excavation_test_data_refrained.csv', 0, 4)

result_list = []
tgt = torch.FloatTensor(1, 3).fill_(-1).to('cuda:0')
for inputs in test_input_list:
    inputs = inputs.squeeze(dim=0).to('cuda:0')
    result = model.generate(inputs, tgt, 0.20)
    result_list.append(result)

test_dataset_answer_list = []
for n in range(4):
    df = pd.read_csv('C:/Users/USER/Programming/Excavator/Data/Excavation_test_answer_' + str(n+1) + '.csv')
    test_dataset_answer = torch.FloatTensor(df.values)
    test_dataset_answer = test_dataset_answer.numpy()
    test_dataset_answer_list.append(test_dataset_answer) 

#print(result_list[0].shape, test_dataset_answer_list[0].shape)
model_score = total_MAPE_score(result_list, test_dataset_answer_list)
print('total score: ', model_score)
