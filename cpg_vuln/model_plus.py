
from __future__ import annotations

from torch import nn
import torch
from torch_geometric.nn import RGCNConv, global_mean_pool, GlobalAttention, Set2Set

from cpg_vuln.domain_knowledge import DOMAIN_GRAPH_DIM


class VulCPGPlus(nn.Module):
    def __init__(
        self,
        num_type_buckets: int,
        num_token_buckets: int,
        num_edge_types: int = 15,
        embed_dim: int = 64,
        hidden_dim: int = 192,
        num_layers: int = 4,
        dropout: float = 0.15,
        use_domain: bool = True,
        node_flag_dim: int = 15,
    ) -> None:
        super().__init__()
        self.use_domain = use_domain
        self.type_emb = nn.Embedding(num_type_buckets, embed_dim)
        self.token_emb = nn.Embedding(num_token_buckets, embed_dim)
        self.flag_proj = nn.Linear(node_flag_dim, embed_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations=num_edge_types))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.att_pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
        self.set2set = Set2Set(hidden_dim, processing_steps=2)

        pool_dim = hidden_dim * 4
        domain_dim = DOMAIN_GRAPH_DIM if use_domain else 0
        self.domain_proj = nn.Linear(DOMAIN_GRAPH_DIM, hidden_dim) if use_domain else None

        head_in = pool_dim + (hidden_dim if use_domain else 0)
        self.bin_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._embed_dim = head_in

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def encode(self, data) -> torch.Tensor:
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
            x = x + h if h.shape == x.shape else h

        g_mean = global_mean_pool(x, batch)
        g_att = self.att_pool(x, batch)
        g_s2s = self.set2set(x, batch)
        g = torch.cat([g_mean, g_att, g_s2s], dim=-1)

        if self.use_domain and self.domain_proj is not None and hasattr(data, "domain"):
            dom = data.domain
            if dom.dim() == 1:
                dom = dom.view(1, -1)
            if dom.size(0) != g.size(0) and dom.numel() == g.size(0) * DOMAIN_GRAPH_DIM:
                dom = dom.view(g.size(0), DOMAIN_GRAPH_DIM)
            g = torch.cat([g, self.domain_proj(dom)], dim=-1)
        elif self.use_domain and self.domain_proj is not None:
            zeros = torch.zeros(g.size(0), DOMAIN_GRAPH_DIM, device=g.device)
            g = torch.cat([g, self.domain_proj(zeros)], dim=-1)

        return g

    def forward(self, data):
        g = self.encode(data)
        logit_bin = self.bin_head(g).squeeze(-1)
        return g, logit_bin, None
