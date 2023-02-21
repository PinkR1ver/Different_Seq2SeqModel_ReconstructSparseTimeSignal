import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.decoder = nn.RNN(input_size=output_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1) # [nstep+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)

        _, h_t = self.encoder(enc_input, enc_hidden)

        outputs, _ = self.decoder(dec_input, h_t)

        model = self.fc(outputs) # [nstep+1, batch_size, n_hidden -> n_class]

        return model

