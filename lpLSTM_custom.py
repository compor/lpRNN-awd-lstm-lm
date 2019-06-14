import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.jit as jit

"""
Reuse code from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
"""
# class lpLSTM(jit.ScriptModule):
class lpLSTM(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    retention_ratio: for low pass filtering the RNN
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, wdropout=0.0
                    ,activation='tanh', train_ret_ratio=False):
        # super(lpLSTMCell, self).__init__(mode='LSTM', input_size=input_size, hidden_size=hidden_size)
        super(lpLSTM, self).__init__()
        print('='*89)
        print('ALERT: Running LSTM Custom module that is not yet fully tested!!!!!!')
        print('='*89)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        if wdropout:
            self.raw_w_ih = th.randn(4 * hidden_size, input_size)
            self.weight_ih = Parameter(F.dropout(self.raw_w_ih, p=wdropout, training=self.training))
            self.raw_w_hh =  th.randn(4 * hidden_size, hidden_size)
            self.weight_hh = Parameter(F.dropout(self.raw_w_hh, p=wdropout, training=self.training))
        else:
            self.weight_ih = Parameter(th.randn(4 * hidden_size, input_size))
            self.weight_hh = Parameter(th.randn(4 * hidden_size, hidden_size))
        
        # Wrap biases as parameters if desired, else as variables without gradients
        self.bias_ih = Parameter(th.randn(4 * hidden_size), requires_grad=self.bias)
        self.bias_hh = Parameter(th.randn(4 * hidden_size), requires_grad=self.bias)

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

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (1, time, input_size, ...)
        outputs = []
        for x in th.unbind(input_, dim=0):
            # print(x.shape, self.w_xi.shape)
            h = self.forward_step(x, hidden)
            outputs.append(h[0].clone())
            hidden = h[1]
        op = th.squeeze(th.stack(outputs))
        # print('MVN', op.shape, hidden[0].shape, hidden[1].shape)
        return op, hidden

    def forward_step(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = th.squeeze(state[0]), th.squeeze(state[1])
        gates = (th.mm(input, self.weight_ih.t()) + self.bias_ih +
                 th.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = th.sigmoid(ingate)
        forgetgate = th.sigmoid(forgetgate)
        cellgate = self.activation(cellgate)
        outgate = th.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.activation(cy)
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
   print(x)
