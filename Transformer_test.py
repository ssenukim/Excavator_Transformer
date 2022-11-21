import torch 
from torch.utils.data import DataLoader, Dataset 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from torch.autograd import Variable 
from Transformer_model import *
from Utility import * 

if __name__ == '__main__':
    CSV_PATH = 'C:/Users/USER/Programming/Excavator/Data/totalExcavationDataR4_cut.csv'    
    
    train_dataset = CustomDataset(CSV_PATH, 0, 1)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name(0))

    model = build_model()

    learning_rate = 0.001
    num_epoch = 120
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()
    
    for x, y in train_dataloader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        
        print(x, x.shape)
        print(y, y.shape)
        

        print(output, output.shape)