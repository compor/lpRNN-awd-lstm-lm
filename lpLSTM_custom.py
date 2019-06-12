import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pdb

class lpLSTM(nn.LSTM):
    def __init__(self, *kargs, **kwargs):
        super(lpLSTM, self).__init__(*kargs, **kwargs)
        self.retention_ratio = nn.Parameter(th.FloatTensor(self.hidden_size).uniform_(0.001, 1), requires_grad=False)


    def forward(self, x, hidden):
        if hidden is None:
            hidden = self._init_hidden(x)
        super(lpLSTM, self).forward(x, hidden)


    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = th.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        result = self.lstm_fwd_batch(input, hx)
        output = result[0]
        hidden = result[1:]
        # pdb.set_trace()
        return output, hidden

    def lstm_fwd_batch(self, input, hx):# self._get_flat_weights(), self.bias, self.dropout, self.training))
        """
        Unpacks the input in batch mode and calls the forward function properly
        """
        outputs = []
        for x in th.unbind(input, dim=0):
            hidden = self.lstm_fwd(x, hx, self._get_flat_weights())#, self.bias, self.dropout, self.training)
            outputs.append(hidden[0].clone())
        return th.stack(outputs, dim=1)

    def lstm_fwd(self, input, hx, weights):
        do_dropout = self.training and self.dropout > 0.0
        hidden_size = self.hidden_size
        w_xi = weights[0][:hidden_size,:].t()
        w_xf = weights[0][hidden_size:2*hidden_size,:].t()
        w_xc = weights[0][2*hidden_size:3*hidden_size,:].t()
        w_xo = weights[0][3*hidden_size:4*hidden_size,:].t()
    
        w_hi = weights[1][:hidden_size,:]
        w_hf = weights[1][hidden_size:2*hidden_size,:]
        w_hc = weights[1][2*hidden_size:3*hidden_size,:]
        w_ho = weights[1][3*hidden_size:4*hidden_size,:]
    
        if self.bias:
            b_xi = weights[2][:hidden_size]
            b_xf = weights[2][hidden_size:2*hidden_size]
            b_xc = weights[2][2*hidden_size:3*hidden_size]
            b_xo = weights[2][3*hidden_size:4*hidden_size]
    
            b_hi = weights[3][:hidden_size]
            b_hf = weights[3][hidden_size:2*hidden_size]
            b_hc = weights[3][2*hidden_size:3*hidden_size]
            b_ho = weights[3][3*hidden_size:4*hidden_size]
    
        h, c = hx
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = input# input.view(input.size(1), -1)

        # Linear mappings
        i_t = th.mm(x, w_xi) + th.mm(h, w_hi)
        f_t = th.mm(x, w_xf) + th.mm(h, w_hf)
        o_t = th.mm(x, w_xo) + th.mm(h, w_ho)
    
        if self.bias:
            i_t += b_xi + b_hi
            f_t += b_xf + b_hf
            o_t += b_xo + b_ho
    
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()

        # cell computations
        c_t = th.mm(x, w_xc) + th.mm(h, w_hc) 
        
        if self.bias:
            c_t += b_xc + b_hc

        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))

        # Reshape for compatibility
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        if self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
   
        return h_t, (h_t, c_t)

if __name__ == '__main__':
   rnn = lpLSTM(input_size=10, hidden_size=20)
   input = th.randn(5, 3, 10)
   h0 = th.randn(1, 3, 20)
   c0 = th.randn(1, 3, 20)
   #output, (hn, cn) = rnn(input, (h0, c0))
   x =  rnn(input, (h0, c0))
   print(len(x))
