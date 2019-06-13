import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V
import torch.jit as jit

class lpLSTM(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    retention_ratio: for low pass filtering the RNN
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0
                    ,activation='tanh', train_ret_ratio=False):
        # super(lpLSTMCell, self).__init__(mode='LSTM', input_size=input_size, hidden_size=hidden_size)
        super(lpLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.weight_ih = Parameter(th.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(th.randn(4 * hidden_size, hidden_size))

        # Wrap biases as parameters if desired, else as variables without gradients
        if bias:
            self.bias_ih = Parameter(th.randn(4 * hidden_size), requires_grad=True)
            self.bias_hh = Parameter(th.randn(4 * hidden_size), requires_grad=True)
        else:
            self.bias_ih = Parameter(th.randn(4 * hidden_size), requires_grad=False)
            self.bias_hh = Parameter(th.randn(4 * hidden_size), requires_grad=False)


        if activation =='tanh':
           self.activation = th.tanh
        else:
           self.activation = th.relu
        self.retention_ratio = nn.Parameter(th.FloatTensor(self.hidden_size).uniform_(0.001, 1)
                                            ,requires_grad=train_ret_ratio)
        layer_params = [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]
        param_names = ['weight_ih_l0', 'weight_hh_l0','bias_ih_l0', 'bias_hh_l0']
        for name, param in zip(param_names, layer_params):
            setattr(self, name, param)
        self.reset_parameters()

    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (1, time, input_size, ...)
        outputs = []
        for x in th.unbind(input_, dim=0):
            # print(x.shape, self.w_xi.shape)
            h = self.forward_single(x, hidden)
            outputs.append(h[0].clone())
            hidden = h[1]
        op = th.squeeze(th.stack(outputs))
        # print('MVN', op.shape, hidden[0].shape, hidden[1].shape)
        return op, hidden

    def forward_single(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (th.mm(input, self.weight_ih.t()) + self.bias_ih +
                 th.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = th.sigmoid(ingate)
        forgetgate = th.sigmoid(forgetgate)
        cellgate = th.tanh(cellgate)
        outgate = th.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * th.tanh(cy)
        # # Filtering 
        hy = self.retention_ratio * hx + (1-self.retention_ratio) * hy
        if self.dropout > 0.0:
            F.dropout(hy, p=self.dropout, training=self.training, inplace=True) 
        return hy, (hy, cy)

if __name__ == '__main__':
   rnn = lpLSTM(input_size=10, hidden_size=20)
   input = th.randn(5, 3, 10)
   h0 = th.randn(1, 3, 20)
   c0 = th.randn(1, 3, 20)
   #output, (hn, cn) = rnn(input, (h0, c0))
   x =  rnn(input, (h0, c0))
   print(len(x))
