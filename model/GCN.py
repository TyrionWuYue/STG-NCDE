import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.
    """

    def __init__(self, model_dim, num_heads=8):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=64, num_heads=8, dropout=0
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        return out


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.dim = dim
        
        # Linear layers for node embeddings
        self.w_dynamic = nn.Linear(in_features=dim, out_features=dim)  # Dynamic embeddings
        self.w_static = nn.Linear(in_features=dim, out_features=dim)   # Static embeddings
        
        # Transformation matrix for static embeddings
        self.t = nn.Parameter(torch.zeros(size=(self.dim, self.dim)))
        nn.init.xavier_uniform_(self.t.data, gain=1.414)
        
        self.norm = nn.LayerNorm(dim)
        
        # Gates
        self.w_r = nn.Linear(in_features=dim, out_features=dim)
        self.u_r = nn.Linear(in_features=dim, out_features=dim)

        self.w_h = nn.Linear(in_features=dim, out_features=dim)
        self.w_u = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, dynamic_nodevec, static_nodevec):
        batch_size = dynamic_nodevec.size(0)
        # Normalize dynamic embeddings
        dynamic_nodevec = self.norm(dynamic_nodevec)
        
        # Process dynamic embeddings
        dynamic_res = self.w_dynamic(dynamic_nodevec) + dynamic_nodevec
        
        # Transform static embeddings
        static_transformed = static_nodevec + torch.einsum('bnd, dd->bnd', [static_nodevec, self.t])
        
        # Gate calculation
        z = torch.sigmoid(dynamic_res + static_transformed)

        r = torch.sigmoid(self.w_r(static_nodevec) + self.u_r(dynamic_nodevec))
        h = torch.tanh(self.w_h(static_nodevec) + r * (self.w_u(dynamic_nodevec)))

        # Final result: weighted sum of static and dynamic embeddings
        res = z * dynamic_nodevec + (1 - z) * h

        return res


# Global GCN (fixed node embeddings)
class GlobalAGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(GlobalAGCN, self).__init__()
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


# Local GCN (Node Embedding change with input data)
class LocalAGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(LocalAGCN, self).__init__()
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.bias, 0)

    def forward(self, x, node_embeddings):
        batch_size, num_nodes, _ = x.size()
        supports = F.softmax(F.relu(torch.matmul(node_embeddings, node_embeddings.transpose(-1,-2))), dim=-1)
        support_set = [torch.eye(num_nodes).to(supports.device).unsqueeze(0).expand(batch_size, -1, -1), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        x_g = []
        for support in support_set:
            x_g.append(torch.einsum("bnm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, cheb_k*C
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv


# Gated Graph Convolution
class GatedGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(GatedGCN, self).__init__()
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.info_fussion = GatedFusion(embed_dim)

        self.global_GCN = GlobalAGCN(dim_in, dim_out, cheb_k, embed_dim) # Global Graph Convolution. (node embeddings is fixed)
        self.local_GCN = LocalAGCN(dim_in, dim_out, cheb_k, embed_dim) # Local Graph Convolution. (node embeddings change with input data)

        # self.gate = nn.Linear(dim_out * 2, dim_out)
        self.out = nn.Linear(dim_out * 2, dim_out)
    
    def forward(self, x, global_E, local_E):
        local_E = self.info_fussion(global_E, local_E)

        global_out = self.global_GCN(x, global_E)
        local_out = self.local_GCN(x, local_E)
        
        combined = torch.cat([global_out, local_out], dim=-1)
        # gate_weights = torch.sigmoid(self.gate(combined))

        # output = gate_weights * global_out + (1 - gate_weights) * local_out
        output = self.out(combined)

        return output