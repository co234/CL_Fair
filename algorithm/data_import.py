import torch
import numpy as np
import pickle
from torch.utils import data

class Dataset():
    def __init__(self, x, labels, sensitive_attribute,clean_label):
        self.x = x
        self.labels = labels
        self.sensitive_attribute = sensitive_attribute
        self.clean_label = clean_label
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return int(index), self.x[index], int(self.labels[index]), int(self.sensitive_attribute[index]), int(self.clean_label[index])
    



def load_data(input_data,target_data,batch_size=64):
    
    train_silver = Dataset(input_data['x'], \
                           input_data['yt'], \
                           input_data['s'], 
                           input_data['y'])
    train_loader = data.DataLoader(train_silver, batch_size=batch_size,shuffle=True)
    

    testing_set = Dataset(target_data['x'], target_data['y'], target_data['s'],target_data['y'])
    test_loader = data.DataLoader(testing_set, batch_size=batch_size,shuffle=True)

    return train_loader, test_loader