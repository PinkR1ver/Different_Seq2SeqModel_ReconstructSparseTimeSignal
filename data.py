import os
from torch.utils.data import Dataset
import torch

import pandas as pd
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, path, window_length, step):
        self.path = path
        self.window_length = window_length
        self.step = step
        self.df = pd.read_csv(self.path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        signal = self.df.loc[index, 'signal']
        signal = signal[1:-1]
        signal = signal.split(',')
        signal = [float(n) for n in signal]
        
        data = []
        i = 0
        signal_length = len(signal)
        while i + self.window_length < signal_length:
            data.append(signal[i:i + self.window_length])
            i += self.step
        
        data = torch.DoubleTensor(data)

        parameter = self.df.loc[index, 'key_parameter']
        parameter = parameter[2:-2]
        parameter = parameter.split('], [')

        parameter_tk = parameter[0].split(',')
        parameter_tk = [float(n) for n in parameter_tk]

        parameter_ak = parameter[1].split(',')
        parameter_ak = [float(n) for n in parameter_ak]

        key_parameter = [parameter_tk, parameter_ak]

        return data, key_parameter


if __name__ == '__main__':
    dataset = SignalDataset('./data.csv', 5, 2)
    print(dataset[10][1])