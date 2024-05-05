import torch
import torch.nn.functional as F
import torch.nn as nn

import math
import time

torch.set_default_dtype(torch.float32)

class Decoder(nn.Module):

    def __init__(self, configs, t_num):
        super().__init__()
        self.input_dim = configs.feature_dim
        self.layers_num = configs.mlp_level
        self.hidden_dim = configs.mlp_hidden_dim
        self.output_dim = configs.mlp_basis_num
        self.t_num = t_num
        self.device = configs.device
        layers = []
        for i in range(self.layers_num):
            if i == 0: 
                layers.append(nn.Linear(self.input_dim, self.hidden_dim, True))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, True))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(self.hidden_dim, self.output_dim, True)

        #DCT initialization
        matrix = torch.ones(self.output_dim, self.t_num, device='cuda')

        for n in range(self.output_dim):
            matrix[n] = (torch.linspace(0, math.pi, self.t_num) + 0.5*math.pi/self.t_num)*n
        dct_matrix = torch.cos(matrix)
        self.basis_matrix_DCT = nn.Parameter(dct_matrix[1:])
        # self.basis_matrix_DCT = dct_matrix[1:]

        # random initialization
        # dct_matrix = torch.randn(self.output_dim-1, self.t_num, device = 'cuda')
        # self.basis_matrix_DCT = nn.Parameter(dct_matrix)

        # Dirac initialization
        # dct_matrix = torch.zeros(self.output_dim-1, self.t_num, device = 'cuda')
        # for i in range(self.t_num):
        #     dct_matrix[i,i] = 1.0
        #     dct_matrix[self.t_num-i-1,i] = 1.0
        # self.basis_matrix_DCT = dct_matrix
        constant_basis = torch.ones(self.t_num,device=self.device).unsqueeze(0)
        self.full_basis = torch.cat((constant_basis, self.basis_matrix_DCT),0)

        self.to(self.device)

    
    def stepforward(self):
        keep_part = self.basis_matrix_DCT[:,1:]
        extend_part = self.basis_matrix_DCT[:,-1].unsqueeze(1)
        self.basis_matrix_DCT = nn.Parameter(torch.cat((keep_part,extend_part),dim=1))
        constant_basis = torch.ones(self.t_num,device=self.device).unsqueeze(0)
        self.full_basis = torch.cat((constant_basis, self.basis_matrix_DCT),0)


    def forward(self, feature, t):
        for k,l in enumerate(self.layers):
            if k==0:
                h = F.relu((l(feature)))
            else:
                h = F.relu(l(h))
        weights = self.lout(h).squeeze(1)

        signals = torch.mm(weights, self.full_basis)
        static_output = weights[:,0]
        dynamic_output = signals[torch.arange(signals.shape[0]),t.reshape(1,-1)].squeeze(0)

        # static_output = weights[:,0]
        # query_time_basis = self.full_basis[:, t.view(-1)].T
        # dynamic_output = (weights*query_time_basis).sum(1)

        return static_output, dynamic_output, weights
    
    
    def interval_max(self, feature, intervals):

        intervals[intervals[:,1] ==-1] = 0.0

        for k,l in enumerate(self.layers):
            if k==0:
                h = F.relu((l(feature)))
            else:
                h = F.relu(l(h))
        weights = self.lout(h).squeeze(1)
        signals = torch.mm(weights, self.full_basis)
        valid_signals = torch.ones_like(signals, dtype=signals.dtype, device=signals.device)*(-100.0)

        start_indices = intervals[:,0].reshape(-1,1)
        end_indieces = intervals[:,1].reshape(-1,1)

        all_indexs = torch.arange(signals.shape[1]).repeat(signals.shape[0],1).to(signals.device)

        mask = (all_indexs >= start_indices) & (all_indexs <= end_indieces)

        valid_signals[mask] = signals[mask]

        valid_max, max_index = torch.max(valid_signals, dim=1)

        return valid_max , max_index
    
    
    def output_curve(self, feature):
        for k,l in enumerate(self.layers):
            if k==0:
                h = F.relu((l(feature)))
            else:
                h = F.relu(l(h))
        weights = self.lout(h).squeeze(1)
        signals = torch.mm(weights, self.full_basis)

        return signals, weights
    
    




