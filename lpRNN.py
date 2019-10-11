import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.jit as jit

"""
Reuse code from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
"""
class lpRNN(nn.Module):
    """
    An implementation of Elman RNN with dropout, weight dropout and low pass filtering added:
    retention_ratio: for low pass filtering the RNN
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, wdropout=0.0
                    ,activation='relu', train_ret_ratio=False, set_retention_ratio=None):
        super(lpRNN, self).__init__()
        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.bias          = bias > 0
        self.dropout       = dropout
        self.wdropout      = wdropout
        self.train_ret_ratio = train_ret_ratio > 0
        if wdropout: #weight dropout
            self.raw_w_ih  = th.randn(hidden_size, input_size)
            self.weight_ih = Parameter(F.dropout(self.raw_w_ih, p=self.wdropout, training=self.training))
            self.raw_w_hh  = th.randn(hidden_size, hidden_size)
            self.weight_hh = Parameter(F.dropout(self.raw_w_hh, p=self.wdropout, training=self.training))
        else:
            self.weight_ih = Parameter(th.randn(hidden_size, input_size))
            self.weight_hh = Parameter(th.randn(hidden_size, hidden_size))
        
        self.bias_ih = Parameter(th.randn(hidden_size), requires_grad=self.bias)
        self.bias_hh = Parameter(th.randn(hidden_size), requires_grad=self.bias)
        # Recurrent activation
        if activation =='tanh':
           self.activation = th.tanh
        else:
           self.activation = th.relu
        # Train low pass filtering factor
        if set_retention_ratio is not None:
            self.retention_ratio = nn.Parameter(set_retention_ratio * th.ones(self.hidden_size)
                                                ,requires_grad=self.train_ret_ratio)
        else:
            self.retention_ratio = nn.Parameter(th.FloatTensor(self.hidden_size).uniform_(0.001, 1)
                                                ,requires_grad=self.train_ret_ratio)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (time_step, batch, input_size, ...)
        outputs = []
        for x in th.unbind(input_, dim=0):
            h = self.forward_step(x, hidden)
            outputs.append(h[0].clone())
            hidden = h[1]
        op = th.squeeze(th.stack(outputs))
        return op, hidden

    def forward_step(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        # ALERT: Bug in code here. Does not work for batch_size of 1.
        hx = th.squeeze(state)
        hy = (th.mm(input, self.weight_ih.t()) + self.bias_ih +
                      th.mm(hx, self.weight_hh.t()) + self.bias_hh)

        hy = self.activation(hy)

        # Filtering 
        hy = self.retention_ratio * hx + (1-self.retention_ratio) * hy
        # hy = self.retention_ratio * hx + hy

        # Dropout
        if self.dropout > 0.0:
            F.dropout(hy, p=self.dropout, training=self.training, inplace=True) 
        return hy, hy

if __name__ == '__main__':
   rnn = lpRNN(input_size=10, hidden_size=20)
   input = th.randn(5, 3, 10)
   h0 = th.randn(1, 3, 20)
   #output, (hn, cn) = rnn(input, (h0, c0))
   x =  rnn(input, (h0))
   print(x[0].shape)
