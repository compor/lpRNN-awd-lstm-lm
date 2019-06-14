import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V

class lpLSTM(nn.Module):
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
    retention_ratio: for low pass filtering the RNN
    """
    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0
                    ,activation='tanh', train_ret_ratio=False):
        # super(lpLSTMCell, self).__init__(mode='LSTM', input_size=input_size, hidden_size=hidden_size)
        super(lpLSTM, self).__init__()
        print('='*89)
        print('ALERT: Running LSTM Custom module that is slow. Use lpLSTM_custom instead!!!!!!')
        print('='*89)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        # input to hidden weights
        self.w_xi = P(T(input_size, hidden_size))
        self.w_xf = P(T(input_size, hidden_size))
        self.w_xo = P(T(input_size, hidden_size))
        self.w_xc = P(T(input_size, hidden_size))
        # hidden to hidden weights
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        # bias terms
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)

        # Wrap biases as parameters if desired, else as variables without gradients
        if bias:
            W = P
        else:
            W = V
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)

        if activation =='tanh':
           self.activation = th.tanh
        else:
           self.activation = th.relu
        self.retention_ratio = nn.Parameter(th.FloatTensor(self.hidden_size).uniform_(0.001, 1)
                                            ,requires_grad=train_ret_ratio)
        layer_params = [self.w_xc, self.w_hc, self.b_i, self.b_c]
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

    def forward_single(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        # x = x.view(x.size(1), -1)
        # Linear mappings
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        # cell computations
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t = self.activation(c_t)
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, self.activation(c_t))

        # Filtering 
        h_t = self.retention_ratio * h + (1-self.retention_ratio) * h_t

        # Reshape for compatibility
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)        
        return h_t, (h_t, c_t)

if __name__ == '__main__':
    model = nn.Sequential(nn.Linear(10, 100), lpLSTM(100,100), lpLSTM(100,100), nn.Linear(100,10))

    print(model)

    params = list(model.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model total parameters:', total_params)