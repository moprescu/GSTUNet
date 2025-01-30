import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SimulatedData(Dataset):
    def __init__(self, data_dir, n_train = 150, train = True, tlen = 1):
        """
        A sample dataset class for causal inference data.
        
        tlen: determines the input time length
        """
        
        # train test split
        if train:
            start, end = 0, n_train
        else:
            start, end = n_train, None
        
        # Loading data while applying the split
        self.X = np.load(os.path.join(data_dir, 'X.npy'))[start:end]
        self.A = np.load(os.path.join(data_dir, 'A.npy'))[start:end]        
        self.Y = np.load(os.path.join(data_dir, 'Y.npy'))[start:end]
        
        self.total_t = len(self.Y)  
        self.tlen = tlen 
        
        if train:                
            self.parse_data(tlen)
            
        else:
            self.parse_data(tlen)
        
    def parse_data(self, tlen):
        """
        tlen:   length of conditional t for predicting Y
        """
        # Get all combinations of data
        self.time_combinations = []
        for i in range(self.total_t - tlen):
            self.time_combinations.append([i+j for j in range(tlen+1)])
            
    def __len__(self):
        return len(self.time_combinations)

    def __getitem__(self, idx):   
        # For every call, return the following data, with random seed idx
        time_indice = np.array(self.time_combinations[idx])
        x, A, Y = torch.tensor(self.X[time_indice]).float(), torch.tensor(self.A[time_indice]).float(), torch.tensor(self.Y[time_indice]).float()
        return x, A, Y