from __future__ import annotations

from torch import nn
import torch
from torch_geometric.nn import RGCNConv, global_mean_pool, GlobalAttention

from cpg_vuln.graph_dataset import NODE_FLAG_DIM


class VulCPGGNN(nn.Module):

    def __init__(
        self,
        num_type_buckets: int,
        num_token_buckets: int,
        num_edge_types: int = 8,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_cwe: int | None = None,
        dropout: float = 0.1,
        flag_dim: int = NODE_FLAG_DIM,
    ) -> None:
        super().__init__()

        self.type_emb = nn.Embedding(num_type_buckets, embed_dim)
        self.token_emb = nn.Embedding(num_token_buckets, embed_dim)

        self.flag_proj = nn.Linear(flag_dim, embed_dim)

        in_dim = embed_dim

        self.num_edge_types = num_edge_types

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(
                    in_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    num_relations=num_edge_types,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.att_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        self.bin_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.cwe_head = None
        if num_cwe is not None:
            self.cwe_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_cwe),
            )

    def forward(self, data):
        type_ids = data.type_ids
        token_ids = data.token_ids
        flags = data.flags
        edge_index, edge_type, batch = data.edge_index, data.edge_type, data.batch

        x = self.type_emb(type_ids) + self.token_emb(token_ids) + self.flag_proj(flags)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_type)
            h = norm(h)
            h = self.act(h)
            h = self.dropout(h)
            if h.shape == x.shape:
                x = x + h
            else:
                x = h

        g_mean = global_mean_pool(x, batch)
        g_att = self.att_pool(x, batch)
        g = torch.cat([g_mean, g_att], dim=-1)

        logit_bin = self.bin_head(g).squeeze(-1)
        logit_cwe = self.cwe_head(g) if self.cwe_head is not None else None

        return g, logit_bin, logit_cwe

