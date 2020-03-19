import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, batch_size, computing_device):
        super(LSTM_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = computing_device

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.hidden = self.init_hidden()

        self.forward_outputs = list()
#         self.hook_layers()

    def init_hidden(self):
        self.hidden = (torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device),
                       torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device))

    def forward(self, x):
        self.hidden = (self.hidden[0], self.hidden[1])
#         chunk = chunk.reshape(1, *chunk.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        output = self.linear(lstm_out[:, -1, :])

        return output

    def hook_layers(self):
        def forward_hook_function(module, tensor_in, tensor_out):
            self.forward_outputs.append(tensor_out)

        for pos, module in self._modules.items():
            if isinstance(module, nn.LSTM):
                module.register_forward_hook(forward_hook_function)


class RNN_Model(nn.Module):
    def __init__(self, encode_dim, hidden_dim, batch_size, chunk_length, computing_device):
        super(RNN_Model, self).__init__()
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.computing_device = computing_device

        self.rnn = nn.RNN(encode_dim, hidden_dim, nonlinearity='tanh', batch_first=True, num_layers=1)
        self.linear = nn.Linear(hidden_dim, encode_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_dim).to(self.computing_device)

    def forward(self, chunk):
        self.hidden = self.hidden.detach()
        chunk = chunk.reshape(1, *chunk.shape)
        rnn_out, self.hidden = self.rnn(chunk, self.hidden)
        output = self.linear(rnn_out)

        return output, self.hidden