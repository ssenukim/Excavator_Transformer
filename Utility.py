import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset 

class CustomDataset(Dataset):
    def __init__(self, csv_path, a, b):
        df = pd.read_csv(csv_path)
        self.input = df.iloc[a:b, 0:7].values 
        self.output = df.iloc[a:b, 7:].values
        self.input = torch.FloatTensor(self.input)
        self.output = torch.FloatTensor(self.output)

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        input_ = torch.FloatTensor(self.input[idx])
        output_ = torch.FloatTensor(self.output[idx])
        return input_, output_

def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluation(y_pred, y_true):
    k = min(y_pred.shape[0], y_true.shape[0])
    y_pred_ = y_pred[:k, :] 
    y_true_ = y_true[:k, :]
    score = MAPEval(y_pred_, y_true_)
    print(score)
    return score

def make_tensor(CSV_PATH):
    df = pandas.read_csv(CSV_PATH)
    data = torch.FloatTensor(df.values)
    return data

def make_input_tensor(CSV_PATH, a, b, device='cuda:0'):
    df = pd.read_csv(CSV_PATH)
    semi_list = []
    test_input_list = []
    
    for n in range(b-a):
        df_2 = df.iloc[n:n+1, :]
        semi_list.append(df_2)
    
    for data in semi_list:
        input_data = torch.FloatTensor(data.values)
        input_data = input_data.unsqueeze(1).to(device)
        test_input_list.append(input_data)
        
    return test_input_list

def testdata_set(*args, y_true):
    score_list = []
    for n, PATH in enumerate(args):
        df = pd.read_csv(PATH)
        test_dataset_answer = torch.FloatTensor(df.values)
        test_dataset_answer = test_dataset_answer.numpy()
        score = evaluation(test_dataset_answer, y_true)
        score_list.append(score)

def total_MAPE_score(pred_list, answer_list):
    if len(pred_list) != len(answer_list):
        raise ValueError('different list size between two parameters')
        
    cnt = 0 
    for k in range(len(pred_list)):
        cnt += evaluation(pred_list[k], answer_list[k])
        
    return cnt / len(pred_list)

        