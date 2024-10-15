import torch
import torch.nn as nn
import torch.nn.functional as F

# 全局图卷积(Node Embedding不变)
class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.constant_(self.bias_pool, 0)

    def forward(self, x, node_embeddings):
        batch_size, num_nodes, _ = x.size()
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(-1,-2))), dim=-1)
        supports_set = [torch.eye(num_nodes).to(x.device), supports]
        supports = torch.stack(supports_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum('knm,bmc->bknc', supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


# Gated Graph Convolution
class GatedGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(GatedGCN, self).__init__()
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.global_GCN = AGCN(dim_in, 2*dim_out, cheb_k, embed_dim) # Global Graph Convolution. (node embeddings is fixed)
        self.local_GCN = AGCN(dim_in, dim_out, cheb_k, embed_dim) # Local Graph Convolution. (node embeddings change with input data)

        self.gate = nn.Linear(dim_out * 2, dim_out)
    
    def forward(self, x, global_E, local_E):
        global_out = self.global_GCN(x, global_E)
        local_out = self.local_GCN(x, local_E)
        
        combined = torch.cat([global_out, local_out])
        gate_weights = torch.sigmoid(self.gate(combined))

        output = gate_weights * global_out + (1 - gate_weights) * local_out

        return output