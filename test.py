
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

    train_dataset = CustomDataset(CSV_PATH, 0, 32)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    for x, y, z in train_dataloader:
        print(x, x.shape)
        print(y, y.shape)
        print(z, z.shape)
