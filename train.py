import torch
import torch.nn as nn
import torch.utils.data as Data
from data import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    window_length = 5
    step = 2
    n_step = 10
    batch_size = 16
    n_hidden = 128
    output_size = 2

    Signal_Dataset =  SignalDataset('./data.csv', window_length=window_length, step=step, n_step=n_step)

    input_size = window_length

    loader = Data.DataLoader(Signal_Dataset, batch_size=16, shuffle=True)

    model = Seq2Seq(input_size=input_size, hidden_size=n_hidden, output_size=output_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):
        for enc_input_batch, dec_input_batch, dec_output_batch in loader:
            # print(enc_input_batch, dec_input_batch, dec_output_batch)

            h_0 = torch.zeros(1, batch_size, n_hidden).to(device)

            enc_input_batch = enc_input_batch.to(device)
            dec_input_batch = dec_input_batch.to(device)
            dec_output_batch = dec_output_batch.to(device)

            pred = model(enc_input_batch, h_0, dec_output_batch)
            
            pred = pred.transpose(0, 1)
            loss = 0

            for i in range(0, len(dec_output_batch)):
                loss += criterion(pred[i], dec_output_batch[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))


    loader = Data.DataLoader(Signal_Dataset, batch_size=1, shuffle=False)
    h_0 = torch.zeros(1, 1, n_hidden).to(device)

    for enc_input_batch, dec_input_batch, dec_output_batch in loader:
        pred = model(enc_input_batch, h_0, dec_input_batch)

        print("Prediction ->  ", pred)
        print("True ->        ", dec_output_batch)
        print(' ')

