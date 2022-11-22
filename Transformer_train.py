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

    train_dataset = CustomDataset(CSV_PATH, 0, 160000)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device, torch.cuda.get_device_name(0))

    model = build_model()
    
    learning_rate = 0.001
    num_epoch = 120
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()

    from tqdm import tqdm
    model.train()
    with tqdm(range(num_epoch)) as tr:
        for i in tr:
            total_loss = 0.0
            for inputs, answer, dec_input in train_dataloader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                answer = answer.to(device)
                dec_input = dec_input.to(device)

                output = model.forward(inputs, dec_input)
                output = output.to(device)
                output = output.squeeze()
                loss = criterion(output, answer)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                tr.set_postfix(loss="{0:.5f}".format(total_loss/len(train_dataloader)))

    model.save('C:/Users/USER/Programming/Excavator/LSTM/Trained_model/' +  'model_1_4lyr.pt', state_dict=False)
    print()
    print('model saved!')