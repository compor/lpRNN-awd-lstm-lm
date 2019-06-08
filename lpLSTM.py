import torch
import torch.nn as nn
import torch.nn.functional as F

class lpLSTM(nn.LSTM):
    def __init__(self, *kargs, **kwargs):
        super(lpLSTM, self).__init__(*kargs, **kwargs)
        self.retention_ratio = nn.Parameter(torch.rand(self.hidden_size), requires_grad=False)
 
    # @weak_script_method
    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        prev_output = hx[0]
        # prev_hidden = hx[1]
        output, hidden = super(lpLSTM, self).forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        output = self.retention_ratio * prev_output + \
            (1 - self.retention_ratio) * output
        return output, hidden