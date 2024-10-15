import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN import GlobalAGCN, GatedGCN, SelfAttentionLayer

class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = z.tanh()
        return z
    

class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        # self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        #                                    for _ in range(num_hidden_layers - 1))

        #FIXME:
        # self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.Wq = nn.Linear(hidden_hidden_channels, embed_dim)
            self.Mem = nn.Parameter(torch.randn(128, embed_dim), requires_grad=True) # 64 is num of memory
            nn.init.xavier_normal_(self.Mem)
            # self.attn = SelfAttentionLayer(hidden_hidden_channels, feed_forward_dim=64, num_heads=8, dropout=0.1)
        
        # self.agcn = AGCN(hidden_hidden_channels, hidden_hidden_channels, cheb_k, embed_dim)
        self.gated_agcn = GatedGCN(hidden_hidden_channels, hidden_hidden_channels, cheb_k, embed_dim)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            # local_emb = self.attn(z)
            query = self.Wq(z)
            att_score = torch.softmax(torch.matmul(query, self.Mem.transpose(-1, -2)), dim=-1)
            local_emb = torch.matmul(att_score, self.Mem)
            # z = self.agcn(z, self.node_embeddings)
            z = self.gated_agcn(z, self.node_embeddings, local_emb)
        else:
            raise ValueError('Check g_type argument')

        #FIXME:
        # z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])