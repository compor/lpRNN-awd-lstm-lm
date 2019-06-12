import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.autograd import Variable as V

class lpLSTMCell(nn.RNNBase):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    Special args:
    dropout_method: one of
            * pytorch: default dropout implementation
            * gal: uses GalLSTM's dropout
            * moon: uses MoonLSTM's dropout
            * semeniuta: uses SemeniutaLSTM's dropout
    """

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0,
                 dropout_method='pytorch', jit=False, activation='relu'):
        super(lpLSTMCell, self).__init__(mode='LSTM', input_size=input_size, hidden_size=hidden_size)
        print('ALERT: Creating unoptimized LSTM cell. Use only if you want relu activation. ')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        assert(dropout_method.lower() in ['pytorch', 'gal', 'moon', 'semeniuta'])
        self.dropout_method = dropout_method
        if activation =='tanh':
           self.activation = F.tanh
        else:
           self.activation = F.relu
        self.retention_ratio = nn.Parameter(th.FloatTensor(self.hidden_size).uniform_(0.001, 1), requires_grad=False)
        # layer_params = [self.i2h.weight, self.h2h.weight, self.i2h.bias, self.h2h.bias]
        # param_names = ['weight_ih_l0', 'weight_hh_l0']
        # if bias:
        #     param_names += ['bias_ih_l0', 'bias_hh_l0']
        # for name, param in zip(param_names, layer_params):
        #     setattr(self, name, param)
        # self._all_weights.append(param_names)
        # for name, p in self.named_parameters():
        #     print(f'Name is {name} with value {p}')
        # for name_w in param_names:
        #     w = getattr(self, name_w)
        #     print(f'w is {w.shape} with name {name_w}')

    def __getattr__(self, attr):
        if attr == 'weight_hh_l0':
            getattr(self.h2h, 'weight')
        elif attr == 'bias_hh_l0':
            getattr(self.h2h, 'bias')
        elif attr == 'weight_ih_l0':
            getattr(self.i2h, 'weight')
        elif attr == 'bias_ih_l0':
            getattr(self.i2h, 'bias')
        else:
            return attr


    def sample_mask(self):
        keep = 1.0 - self.dropout
        self.mask = V(th.bernoulli(T(1, self.hidden_size).fill_(keep)))

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        if hidden is None:
            hidden = self._init_hidden(x)
        do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(1), -1)

        # Linear mappings
        preact = self.i2h(x) + self.h2h(h)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:]#.tanh()
        g_t = self.activation(g_t) # apply activation
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        if do_dropout and self.dropout_method == 'semeniuta':
            g_t = F.dropout(g_t, p=self.dropout, training=self.training)

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        if do_dropout and self.dropout_method == 'moon':
            c_t.data.set_(th.mul(c_t, self.mask).data)
            c_t.data *= 1.0 / (1.0 - self.dropout)

        c_t = self.activation(c_t) # apply activation
        h_t = th.mul(o_t, c_t)#.tanh())

        # Filtering 
        h_t = self.retention_ratio * h + (1-self.retention_ratio) * h_t

        # Reshape for compatibility
        if do_dropout:
            if self.dropout_method == 'pytorch':
                F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == 'gal':
                h_t.data.set_(th.mul(h_t, self.mask).data)
                h_t.data *= 1.0 / (1.0 - self.dropout)

        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        
        return h_t, (h_t, c_t)

class lpLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, activation='relu'):
        super().__init__()
        self.lstm_cell = lpLSTMCell(input_size, hidden_size, bias, activation=activation)

    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (1, time, input_size, ...)

        outputs = []
        for x in th.unbind(input_, dim=1):
            hidden = self.lstm_cell(x, hidden)
            outputs.append(hidden[0].clone())

        return th.stack(outputs, dim=1)
