import torch
import torch.nn as nn
import torch.nn.functional as F

class lpLSTM(nn.LSTM):
    def __init__(self, *kargs, **kwargs):
        super(lpLSTM, self).__init__(*kargs, **kwargs)
        # self.retention_ratio = nn.Parameter(torch.rand(self.hidden_size), requires_grad=False)
        self.retention_ratio = nn.Parameter(torch.FloatTensor(self.hidden_size).uniform_(0.001, 1))#, requires_grad=True)
        # self.retention_ratio = nn.Parameter(torch.FloatTensor(self.hidden_size).uniform_(0.1, 0.2), requires_grad=False)
        # self.retention_ratio = nn.Parameter(torch.FloatTensor(self.hidden_size).normal_(), requires_grad=False)

    # @weak_script_method
    def forward_impl(self, input, hx, batch_sizes, max_batch_size, sorted_indices):
        prev_output = hx[0]
        # prev_hidden = hx[1]
        output, hidden = super(lpLSTM, self).forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        output = self.retention_ratio * prev_output + \
            (1 - self.retention_ratio) * output
        return output, hidden

if __name__ == '__main__':
    model = nn.Sequential(nn.Linear(10, 100), lpLSTM(100,100), lpLSTM(100,100), nn.Linear(100,10))
    print(model)

    params = list(model.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model total parameters:', total_params)