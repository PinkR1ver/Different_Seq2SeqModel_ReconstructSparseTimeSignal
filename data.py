import os
from torch.utils.data import Dataset
import torch

import pandas as pd
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, path, window_length, step, n_step):
        self.path = path
        self.window_length = window_length
        self.step = step
        self.df = pd.read_csv(self.path)
        self.n_step = n_step

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

        enc_input_all = torch.FloatTensor(data)

        parameter = self.df.loc[index, 'key_parameter']
        parameter = parameter[2:-2]
        parameter = parameter.split('], [')

        parameter_tk = parameter[0].split(',')
        parameter_tk = [float(n) for n in parameter_tk]

        parameter_ak = parameter[1].split(',')
        parameter_ak = [float(n) for n in parameter_ak]

        key_parameter = [parameter_tk, parameter_ak]

        key_parameter_combine = []

        for i in range(len(key_parameter[0])):
            key_parameter_combine.append((key_parameter[0][i], key_parameter[1][i]))

        for i in range(len(key_parameter_combine), self.n_step):
            key_parameter_combine.append((0 ,0))

        key_parameter_combine.insert(0, (-255, -255))
        dec_input_all = torch.FloatTensor(key_parameter_combine)

        key_parameter_combine.pop(0)
        key_parameter_combine.append((255, 255))
        dec_output_all = torch.FloatTensor(key_parameter_combine)

        return enc_input_all, dec_input_all, dec_output_all

if __name__ == '__main__':
    dataset = SignalDataset('./data.csv', 5, 2, 10)
    print(dataset[15][0])